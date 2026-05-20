"""TDD tests for nested-logit instrument validity (audit findings 1-3).

Each finding has two tests:
- ``violated_invariant``: encodes the bug. PASSES on buggy code, FAILS after fix.
- ``honest_fix``: encodes the faithful target. FAILS on buggy code, PASSES after fix.

The tests build the estimation panel by calling the data-generation pipeline
directly, so importing ``run`` never triggers ``main()`` (guarded by
``if __name__ == '__main__'``).
"""
import sys
from pathlib import Path

import pytest

_TUT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TUT_DIR))

import run  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    """Full estimation panel: products, true shares, instruments."""
    df = run.generate_product_data(n_markets=50)
    df = run.compute_true_shares(df)
    df = run.generate_instruments(df)
    return df


# ---------------------------------------------------------------------------
# Finding 1: "Rival sugar, all products" instrument
# ---------------------------------------------------------------------------

def test_finding1_violated_invariant_rival_sugar_constant(panel):
    """Bug: rival_sugar_sum has zero within-product cross-market variation.

    PASSES on buggy code; FAILS after the honest fix.
    """
    within_product_std = panel.groupby("product_id")["rival_sugar_sum"].std()
    assert within_product_std.max() == 0.0


def test_finding1_honest_fix_rival_sugar_varies(panel):
    """Honest fix: rival_sugar_sum varies across markets within every product.

    FAILS on buggy code; PASSES after the honest fix.
    """
    within_product_std = panel.groupby("product_id")["rival_sugar_sum"].std()
    assert within_product_std.min() > 0.0


# ---------------------------------------------------------------------------
# Finding 2: "Number of products in nest" instrument -> replaced
# ---------------------------------------------------------------------------

def test_finding2_violated_invariant_num_in_nest_constant(panel):
    """Bug: num_in_nest is identically 2 for every observation.

    PASSES on buggy code; FAILS after the honest fix (column dropped or varying).
    """
    assert "num_in_nest" in panel.columns
    assert panel["num_in_nest"].nunique() == 1


def test_finding2_honest_fix_same_nest_cost_iv_varies(panel):
    """Honest fix: a market-varying same-nest cost instrument replaces num_in_nest.

    FAILS on buggy code (column absent); PASSES after the honest fix.
    """
    assert "same_nest_rival_cost" in panel.columns
    within_product_std = panel.groupby("product_id")["same_nest_rival_cost"].std()
    assert within_product_std.min() > 0.0


# ---------------------------------------------------------------------------
# Finding 3: "Same-nest rival sugar" instrument
# ---------------------------------------------------------------------------

def test_finding3_violated_invariant_same_nest_sugar_constant(panel):
    """Bug: same_nest_rival_sugar has zero within-product cross-market variation.

    PASSES on buggy code; FAILS after the honest fix.
    """
    within_product_std = panel.groupby("product_id")["same_nest_rival_sugar"].std()
    assert within_product_std.max() == 0.0


def test_finding3_honest_fix_same_nest_sugar_varies(panel):
    """Honest fix: same_nest_rival_sugar varies across markets within every product.

    FAILS on buggy code; PASSES after the honest fix.
    """
    within_product_std = panel.groupby("product_id")["same_nest_rival_sugar"].std()
    assert within_product_std.min() > 0.0


# ---------------------------------------------------------------------------
# Order condition: 2 endogenous regressors need >= 2 excluded instruments
# that actually vary. The honest fix must satisfy rank/order identification.
# ---------------------------------------------------------------------------

def test_order_condition_excluded_instruments_vary(panel):
    """Honest fix: at least two excluded instruments for ln(s_{j|g}) vary.

    FAILS on buggy code (all three nest instruments degenerate);
    PASSES after the honest fix.
    """
    candidates = ["same_nest_rival_sugar", "same_nest_rival_cost"]
    present = [c for c in candidates if c in panel.columns]
    varying = [
        c for c in present
        if panel.groupby("product_id")[c].std().min() > 0.0
    ]
    assert len(varying) >= 2
