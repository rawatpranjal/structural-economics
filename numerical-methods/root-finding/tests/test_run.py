"""Faithfulness regression tests for the root-finding tutorial.

Derived from bullshit-detector_root-finding_2026-05-20.md. Each audit finding
contributes two tests:

* a *violated-invariant* test that passes on the buggy code (it pins the bug);
* an *honest-fix* test that fails on the buggy code and passes after the fix.

After the fix lands, the violated-invariant tests are expected to fail and the
honest-fix tests are expected to pass. They are kept (xfail-marked where they
pin the now-removed bug) so the regression intent stays readable.

``run.py`` guards ``main()`` behind ``__name__ == "__main__"``, so importing it
only defines the solver functions; no figures or tables are regenerated.
"""

import csv
import importlib.util
from pathlib import Path

import pytest

TUT_DIR = Path(__file__).resolve().parents[1]
RUN_PY = TUT_DIR / "run.py"
README = TUT_DIR / "README.md"
CSV_PATH = TUT_DIR / "tables" / "comparison.csv"


def _load_run_module():
    spec = importlib.util.spec_from_file_location("root_finding_run", RUN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iteration_counts():
    """Iteration count per method from the committed comparison table."""
    counts = {}
    with open(CSV_PATH, newline="") as fh:
        for row in csv.DictReader(fh):
            counts[row["Method"]] = int(row["Iterations"])
    return counts


def _run_source():
    return RUN_PY.read_text()


def _readme_text():
    return README.read_text()


# ---------------------------------------------------------------------------
# Finding 1: "roughly an order of magnitude fewer iterations than bisection"
# is FALSE. Table ratios are 29/7 = 4.1x and 29/5 = 5.8x, not 10x.
# ---------------------------------------------------------------------------

def test_iteration_ratios_are_not_an_order_of_magnitude():
    """Violated invariant: the audit shows the ratios are 4-6x, well below 10x.

    This passes on both buggy and fixed code; it pins the factual basis of the
    finding (the table ratios). The buggy prose contradicts this; the fixed
    prose agrees with it.
    """
    counts = _iteration_counts()
    bisection = counts["Bisection"]
    brent_ratio = bisection / counts["Brent"]
    newton_ratio = bisection / counts["Newton-Raphson"]
    assert 4.0 <= brent_ratio < 10.0
    assert 4.0 <= newton_ratio < 10.0


@pytest.mark.xfail(reason="pins the buggy prose; expected to fail after the fix",
                   strict=True)
def test_prose_still_claims_order_of_magnitude():
    """Violated invariant: buggy README claims 'order of magnitude fewer'."""
    assert "order of magnitude fewer iterations than bisection" in _readme_text()


def test_prose_does_not_overstate_iteration_ratio():
    """Honest fix: README must not claim an order-of-magnitude speedup.

    Fails on buggy code (the false claim is present), passes after the fix.
    """
    text = _readme_text()
    assert "order of magnitude fewer iterations than bisection" not in text


def test_prose_states_the_true_ratio():
    """Honest fix: the corrected sentence names a 4-6x speedup.

    Fails on buggy code (no such phrasing), passes after the fix.
    """
    text = _readme_text()
    assert "4-6x" in text or "four to six" in text


# ---------------------------------------------------------------------------
# Finding 2: "hand-coded Brent root matches scipy.optimize.brentq to 0.00e+00".
# Recompute the residual directly from the solver functions on the tutorial's
# own calibration; it must round to 0.00e+00 at the tutorial's print format.
# ---------------------------------------------------------------------------

def _equilibrium_setup():
    """Reproduce the tutorial's calibration and excess-demand function."""
    alpha, beta, delta = 0.36, 0.96, 0.08
    r_star = 1.0 / beta - 1.0
    k_target = (alpha / (r_star + delta)) ** (1.0 / (1.0 - alpha))

    def Z(r):
        return (alpha / (r + delta)) ** (1.0 / (1.0 - alpha)) - k_target

    return Z


def test_brent_matches_scipy_brentq():
    """Honest fix / re-run check: hand-coded Brent matches scipy.brentq.

    The audit could not ground 0.00e+00 because it is not persisted. This
    recomputes it from run.py's own solver. Passes on both buggy and fixed
    code (the value was always correct); it grounds the README claim.
    """
    from scipy.optimize import brentq

    module = _load_run_module()
    Z = _equilibrium_setup()
    tol = 1e-10
    a0, b0 = 0.0 + 1e-6, 0.10
    bre_root, _ = module.brent_with_history(Z, a0, b0, tol, max_iter=200)
    scipy_root = brentq(Z, a0, b0, xtol=tol)
    residual = abs(bre_root - scipy_root)
    # README prints with ":.2e"; the claim is the formatted value is 0.00e+00.
    assert f"{residual:.2e}" == "0.00e+00"


def test_scipy_match_value_persisted():
    """Honest fix: the Brent-vs-scipy residual is written to a CSV.

    Fails on buggy code (no persisted artifact), passes after the fix adds
    tables/scipy_match.csv so future audits can ground the README value.
    """
    match_csv = TUT_DIR / "tables" / "scipy_match.csv"
    assert match_csv.exists()
    with open(match_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert "0.00e+00" in _readme_text()
    assert rows[0]["brent_minus_scipy"] == "0.00e+00"


# ---------------------------------------------------------------------------
# Finding 3: Equations numbers Newton as Method 3 and Brent as Method 4, but
# Solution Method and the results table both order Brent before Newton.
# ---------------------------------------------------------------------------

def test_methods_list_orders_brent_before_newton():
    """Violated invariant: the methods list (the canonical order) is
    Bisection, Secant, Brent, Newton-Raphson.

    Passes on both buggy and fixed code; it pins the order Equations must match.
    """
    counts_order = []
    with open(CSV_PATH, newline="") as fh:
        for row in csv.DictReader(fh):
            counts_order.append(row["Method"])
    assert counts_order == ["Bisection", "Secant", "Brent", "Newton-Raphson"]


@pytest.mark.xfail(reason="pins the buggy Equations numbering; fails after fix",
                   strict=True)
def test_equations_still_numbers_newton_third():
    """Violated invariant: buggy README labels Newton as Method 3."""
    assert "### Method 3: Newton-Raphson" in _readme_text()


def test_equations_numbers_brent_third_and_newton_fourth():
    """Honest fix: Equations numbering matches the table order.

    Fails on buggy code (Newton is Method 3), passes after the fix.
    """
    text = _readme_text()
    assert "### Method 3: Brent" in text
    assert "### Method 4: Newton-Raphson" in text
    assert "### Method 3: Newton-Raphson" not in text
    assert "### Method 4: Brent" not in text
