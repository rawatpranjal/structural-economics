"""Faithfulness tests for the production-functions-markups tutorial.

From bullshit-detector_production-functions-markups_2026-05-20.md:

Finding 1 (DILUTED, MED): `material_share` was generated directly from
`true_markup`, not read off the panel. The Equations section defines it as
materials expenditure over revenue.

Finding 2 (DILUTED, MED): the proxy inversion hardcoded the true investment-
schedule coefficients (0.75, 0.20, 0.90), an oracle inversion, while the README
implies a nonparametric h^{-1} in the Olley-Pakes / Levinsohn-Petrin tradition.

Finding 3 (DILUTED, LOW): the Results prose attributed all OLS bias to
flexible-input simultaneity. Capital is a predetermined state, not a flexible
input, and is not de-biased by a single-stage proxy control.

Each finding gets a violated-invariant test (passed on the buggy code) and an
honest-fix test (failed on the buggy code). After the fixes the violated
invariants fail and the honest fixes pass.

run.py runs only under `if __name__ == "__main__"`, so importing it is safe.
The estimator tests inspect function source text where the audit specified it.
"""
import importlib.util
import inspect
from pathlib import Path

import numpy as np

TUTORIAL_DIR = Path(__file__).resolve().parents[1]


def _load_run_module():
    spec = importlib.util.spec_from_file_location(
        "production_functions_run", TUTORIAL_DIR / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _readme_text():
    return (TUTORIAL_DIR / "README.md").read_text()


# -------------------------------------------------------------------------
# Finding 1: material_share generated from true_markup vs read off the panel
# -------------------------------------------------------------------------

def test_finding1_violated_invariant_share_generated_from_true_markup():
    """Violated invariant: `simulate_panel` builds `material_share` directly
    from `TRUE_BETA["Materials"] / true_markup`, not from the panel's
    materials and output columns.

    PASSED on the buggy code; FAILS after the fix.
    """
    module = _load_run_module()
    src = inspect.getsource(module.simulate_panel)
    share_line = next(
        line for line in src.splitlines() if "material_share =" in line
    )
    assert "true_markup" in share_line


def test_finding1_honest_fix_share_read_off_the_panel():
    """Honest fix: `material_share` equals the panel's materials revenue share,
    exp(log_materials) / exp(log_output), with unit prices.

    FAILED on the buggy code; PASSES after the fix.
    """
    module = _load_run_module()
    df = module.simulate_panel()
    panel_share = np.exp(df["log_materials"]) / np.exp(df["log_output"])
    assert abs(df["material_share"].mean() - panel_share.mean()) < 0.05


# -------------------------------------------------------------------------
# Finding 2: oracle inversion vs nonparametric inversion
# -------------------------------------------------------------------------

def test_finding2_violated_invariant_oracle_coefficients_hardcoded():
    """Violated invariant: `estimate_production` hardcodes the true
    investment-schedule coefficients 0.75 and 0.90 in the inversion.

    PASSED on the buggy code; FAILS after the fix.
    """
    module = _load_run_module()
    src = inspect.getsource(module.estimate_production)
    assert "(inv - 0.75 - 0.20 * k) / 0.90" in src


def test_finding2_honest_fix_nonparametric_inversion():
    """Honest fix: the productivity control is built nonparametrically with
    np.polyfit, using no true investment-schedule coefficient.

    FAILED on the buggy code; PASSES after the fix.
    """
    module = _load_run_module()
    src = inspect.getsource(module.estimate_production)
    assert "polyfit" in src
    assert "(inv - 0.75 - 0.20 * k) / 0.90" not in src


# -------------------------------------------------------------------------
# Finding 3: capital is predetermined, not a flexible input
# -------------------------------------------------------------------------

def test_finding3_violated_invariant_prose_omits_capital_bias_channel():
    """Violated invariant: the production-estimates figure description blamed
    OLS bias on flexible-input choice without naming capital as a separate,
    predetermined input.

    PASSED on the buggy code (the README never tied capital to "predetermined");
    FAILS after the fix.
    """
    readme = _readme_text()
    assert "predetermined" not in readme


def test_finding3_honest_fix_prose_explains_capital_separately():
    """Honest fix: the README explains that capital is a predetermined state
    and that the single-stage proxy control does not identify its elasticity.

    FAILED on the buggy code; PASSES after the fix.
    """
    readme = _readme_text()
    assert "predetermined" in readme
    assert "capital" in readme.lower()
