"""Faithfulness tests for the merger-simulation tutorial.

From bullshit-detector_merger-simulation_2026-05-20.md, Finding 1
(DILUTED / HIGH, result-changing): `calibrate_logit` omitted the observed-price
factor from its single-product alpha estimate, so the calibrated logit price
coefficient was not the consistent single-product estimate and the posted logit
average price increase was wrong (11.15% instead of 12.79%).

The single-product logit FOC for product j is
    0 = s_j + (p_j - c_j) * alpha * s_j * (1 - s_j),
and with p_j - c_j = m_j * p_j it gives alpha = -1 / (m_j * p_j * (1 - s_j)).
The buggy code dropped the p_j factor.

Each finding gets a violated-invariant test (passed on the buggy code) and an
honest-fix test (failed on the buggy code). After the fix the violated
invariant fails and the honest fix passes.

run.py guards execution behind `if __name__ == "__main__"`, so importing it
does not run the pipeline. The alpha-formula test inspects the function source
text so it is robust to which inputs are passed.
"""
import importlib.util
import inspect
from pathlib import Path

import numpy as np

TUTORIAL_DIR = Path(__file__).resolve().parents[1]

# The Part-3 six-product calibration inputs, copied verbatim from run.py main().
SHARES_OBS = np.array([0.12, 0.10, 0.15, 0.13, 0.08, 0.07])
PRICES_OBS = np.array([1.0, 1.2, 0.9, 1.1, 1.3, 1.4])
MARGINS_OBS = np.array([0.40, 0.35, 0.45, 0.40, 0.30, 0.28])
P2F_PRE = np.array([1, 1, 2, 2, 3, 3])


def _load_run_module():
    spec = importlib.util.spec_from_file_location(
        "merger_simulation_run", TUTORIAL_DIR / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -------------------------------------------------------------------------
# Finding 1: calibrate_logit omits the price factor from the alpha formula
# -------------------------------------------------------------------------

def test_finding1_violated_invariant_alpha_formula_drops_price_factor():
    """Violated invariant: the alpha estimate uses the denominator
    `margins_obs * (1.0 - shares_obs)`, with no `prices_obs` factor.

    PASSED on the buggy code; FAILS after the fix.
    """
    module = _load_run_module()
    src = inspect.getsource(module.calibrate_logit)
    assert "-1.0 / (margins_obs * (1.0 - shares_obs))" in src


def test_finding1_honest_fix_alpha_formula_keeps_price_factor():
    """Honest fix: the alpha estimate keeps the observed-price factor, so it is
    the consistent single-product estimate, and the calibrated coefficient
    equals the average of `-1 / (m_j * p_j * (1 - s_j))`.

    FAILED on the buggy code; PASSES after the fix.
    """
    module = _load_run_module()
    src = inspect.getsource(module.calibrate_logit)
    assert "margins_obs * prices_obs * (1.0 - shares_obs)" in src

    cal = module.calibrate_logit(SHARES_OBS, PRICES_OBS, MARGINS_OBS, P2F_PRE)
    expected_alpha = float(
        np.mean(-1.0 / (MARGINS_OBS * PRICES_OBS * (1.0 - SHARES_OBS)))
    )
    assert abs(cal["alpha"] - expected_alpha) < 1e-9


def test_finding1_honest_fix_logit_avg_price_increase():
    """Honest fix: with the corrected alpha, the logit average price increase
    for the four merging products is 12.79%, not the 11.15% the buggy code
    posted (a 14.7% relative shift in the lead result).

    FAILED on the buggy code; PASSES after the fix.
    """
    import scipy.optimize

    module = _load_run_module()
    p2f_post = np.array([1, 1, 1, 1, 3, 3])
    omega_post = module.ownership_matrix(p2f_post)
    cal = module.calibrate_logit(SHARES_OBS, PRICES_OBS, MARGINS_OBS, P2F_PRE)
    p_post = scipy.optimize.fsolve(
        module.foc_logit, x0=PRICES_OBS * 1.05,
        args=(cal["mc"], cal["alpha"], cal["xi"], omega_post),
    )
    dp = (p_post - PRICES_OBS) / PRICES_OBS * 100.0
    avg_dp = float(np.mean(dp[:4]))
    assert abs(avg_dp - 12.79) < 0.1
