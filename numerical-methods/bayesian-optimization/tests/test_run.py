"""Faithfulness regression tests for the bayesian-optimization tutorial.

Each finding from bullshit-detector_bayesian-optimization_2026-05-20.md gets a
pair of tests:

- a claim-as-invariant test that asserts the audited claim verbatim, and
- an honest-fix test that asserts the faithful state.

All three findings are prose/data mismatches: the code and the comparison
table were already correct, only the README prose overstated them. So the
claim-as-invariant test FAILS on the buggy code (the data already contradicts
the false prose) and stays failing after the fix (the prose no longer makes
the false claim). The honest-fix test FAILS on the buggy code (the false
phrase is still present) and PASSES after the fix. The honest-fix tests are
the regression gate; the claim-as-invariant tests document why the claim is
false against the artifacts.

run.py guards execution behind `if __name__ == "__main__"`, so importing it
does not run the whole tutorial. Prose lives in ModelReport strings inside
run.py, so prose findings are checked against the run.py source text.
"""

import csv
import importlib.util
import inspect
import math
from pathlib import Path

import numpy as np

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
RUN_PY = TUTORIAL_DIR / "run.py"
COMPARISON_CSV = TUTORIAL_DIR / "tables" / "method_comparison.csv"


def _run_source() -> str:
    return RUN_PY.read_text()


def _load_run_module():
    spec = importlib.util.spec_from_file_location("bo_run", RUN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _eval_counts() -> dict[str, int]:
    """Return {method: function-evaluation count} from the comparison table."""
    with COMPARISON_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    return {row["Method"]: int(row["Function evaluations"]) for row in rows}


# =============================================================================
# Finding 1: zero-mean prior claim vs constant-mean GP implementation
# =============================================================================
def test_f1_claim_zero_mean_phrase_present():
    """Claim-as-invariant: the audited 'zero-mean prior' phrase is the bug. It
    PASSES on the buggy run.py source (the mislabelled phrase is in the
    Equations section) and FAILS after the fix (the phrase is removed).

    The code fact behind the audit: fit() centres targets on their sample
    mean and predict() adds that mean back, so the GP has a non-zero constant
    mean. The 'zero-mean' Equations label contradicts that code.
    """
    module = _load_run_module()
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 5.0, 8)
    y = 100.0 + rng.normal(size=8)  # non-zero-mean targets
    gp = module.GaussianProcess().fit(X, y)
    assert gp.y_mean != 0.0  # the code fact behind the audit
    assert "zero-mean" in _run_source()


def test_f1_code_is_constant_mean_gp():
    """The faithful state: fit() centres targets on their sample mean, and
    predict() adds that mean back. This is a constant-mean GP."""
    module = _load_run_module()
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 5.0, 8)
    y = 100.0 + rng.normal(size=8)
    gp = module.GaussianProcess().fit(X, y)
    assert gp.y_mean == float(np.mean(y))


def test_f1_honest_fix_equations_describe_constant_mean():
    """Honest fix: the Equations section must not teach a zero-mean prior.

    The code implements a constant-mean GP (mu = y_mean + k_s^T alpha). The
    Equations prose must agree: it must not contain the phrase "zero-mean" and
    the posterior-mean formula must center the targets on the mean.
    """
    source = _run_source()
    assert "zero-mean" not in source
    # posterior mean must show the constant-mean form: m(X) added back and
    # targets centered as (y - m(X)).
    assert "y - m(X)" in source


# =============================================================================
# Finding 2: "two orders of magnitude" vs the actual 33.6x ratio
# =============================================================================
def test_f2_claim_two_orders_phrase_present():
    """Claim-as-invariant: the audited 'two orders of magnitude' phrase is the
    bug. It PASSES on the buggy run.py source (the false phrase is hardcoded)
    and FAILS after the fix (the phrase is removed).

    The data also contradicts the phrase: 1007 / 30 = 33.6x, log10 = 1.53,
    short of the 2.0 a 'two orders' label requires.
    """
    counts = _eval_counts()
    bo = counts["Bayesian optimization (EI)"]
    sa = counts["Simulated annealing"]
    assert math.log10(sa / bo) < 2.0  # the data fact behind the audit
    assert "two orders of magnitude" in _run_source()


def test_f2_honest_fix_ratio_is_about_1p5_orders():
    """Honest fix: the SA/BO ratio is ~1.5 orders, and the prose must not
    overstate it as 'two orders of magnitude'."""
    counts = _eval_counts()
    bo = counts["Bayesian optimization (EI)"]
    sa = counts["Simulated annealing"]
    assert 1.4 <= math.log10(sa / bo) <= 1.7
    source = _run_source()
    assert "two orders of magnitude" not in source


# =============================================================================
# Finding 3: "one order smaller than random search" vs the actual 16.7x ratio
# =============================================================================
def test_f3_claim_one_order_phrase_present():
    """Claim-as-invariant: the audited 'one order smaller than random search'
    phrase is the bug. It PASSES on the buggy run.py source (the false phrase
    is hardcoded) and FAILS after the fix (the phrase is removed).

    The data also contradicts the phrase: 500 / 30 = 16.7x, log10 = 1.22,
    above the 1.0 a 'one order' label implies.
    """
    counts = _eval_counts()
    bo = counts["Bayesian optimization (EI)"]
    rs = counts["Random search"]
    assert math.log10(rs / bo) > 1.1  # the data fact behind the audit
    assert "one order smaller than random search" in _run_source()


def test_f3_honest_fix_random_search_ratio_above_one_order():
    """Honest fix: RS/BO is above one order; the prose must not flatten the
    random-search and multi-start ratios into a single 'one order' label."""
    counts = _eval_counts()
    bo = counts["Bayesian optimization (EI)"]
    rs = counts["Random search"]
    assert math.log10(rs / bo) > 1.1
    source = _run_source()
    assert "one order smaller than random search" not in source
