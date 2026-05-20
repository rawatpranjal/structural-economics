"""Faithfulness tests for the heaton-lucas tutorial.

Each finding from bullshit-detector_heaton-lucas_2026-05-20.md gets two tests:
a violated-invariant test (passes on the original buggy code) and an
honest-fix test (fails on the original buggy code, passes after the fix).

run.py executes the whole tutorial on import, so these tests inspect the
run.py source text and the generated table artifacts instead of importing it.
"""
import re
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
RUN_PY = HERE.parent / "run.py"
TABLES = HERE.parent / "tables"
SRC = RUN_PY.read_text()


def _residual_block() -> str:
    """Return the body of make_residual_fn from run.py source."""
    start = SRC.index("def make_residual_fn")
    end = SRC.index("# Compile JAX functions")
    return SRC[start:end]


# ---------------------------------------------------------------------------
# Finding 1: KKT non-negativity mu_i >= 0 stated but not enforced.
# ---------------------------------------------------------------------------
def test_f1_violated_invariant_product_complementarity():
    """Original code used the product form mu * constraint = 0.

    The product form does not enforce mu >= 0, so this test documents the
    buggy state. It FAILS once the honest min-formulation fix lands.
    """
    block = _residual_block()
    product_form = bool(re.search(r"\bms1\s*\*\s*s1p\b", block))
    assert product_form, "expected the original product complementarity form"


def test_f1_honest_fix_min_complementarity():
    """An honest fix enforces mu >= 0 via the min(mu, constraint) formulation.

    min(mu, c) = 0 holds iff mu >= 0, c >= 0, and mu * c = 0, which is the
    full Kuhn-Tucker condition the README states.
    """
    block = _residual_block()
    uses_min = "jnp.minimum" in block
    no_product = not re.search(r"\bms1\s*\*\s*s1p\b", block)
    assert uses_min and no_product, (
        "expected jnp.minimum complementarity and no product form"
    )


# ---------------------------------------------------------------------------
# Finding 2: Euler-error table uses 4 paths, not the stated 24.
# ---------------------------------------------------------------------------
def test_f2_violated_invariant_hardcoded_four_paths():
    """Original code capped the Euler-error loop at 4 paths via min(n_paths, 4)."""
    assert "min(n_paths, 4)" in SRC


def test_f2_honest_fix_full_path_count():
    """An honest fix removes the hardcoded 4-path cap on the Euler-error loop."""
    assert "min(n_paths, 4)" not in SRC


# ---------------------------------------------------------------------------
# Findings 3-6: runtime scalars asserted in README but not archived.
# ---------------------------------------------------------------------------
def test_f3to6_violated_invariant_no_scalars_csv():
    """Original repo had no scalars.csv backing the README numbers.

    This test FAILS once the honest fix writes tables/scalars.csv, which is the
    intended outcome.
    """
    block = _residual_block()  # touch source so the test is meaningful
    assert block
    assert not (TABLES / "scalars.csv").exists() or "scalars.csv" not in SRC


def test_f3to6_honest_fix_scalars_csv_matches_readme():
    """An honest fix archives every README scalar to tables/scalars.csv.

    The README is generated from run.py, so the committed README numbers and
    the archived CSV come from the same run and must agree.
    """
    csv = TABLES / "scalars.csv"
    assert csv.exists(), "expected tables/scalars.csv"
    df = pd.read_csv(csv)
    keys = set(df["metric"])
    required = {
        "eq_premium_min_pct", "eq_premium_max_pct",
        "omega_mean", "omega_p10", "omega_p90",
        "no_short_share_pct", "borrow_share_pct",
        "final_policy_change", "max_pointwise_residual",
    }
    assert required <= keys, f"missing scalars: {required - keys}"

    readme = (HERE.parent / "README.md").read_text()
    val = {r["metric"]: float(r["value"]) for _, r in df.iterrows()}
    # The equity-premium range sentence must quote the archived numbers.
    assert f"{val['eq_premium_min_pct']:.2f}%" in readme
    assert f"{val['eq_premium_max_pct']:.2f}%" in readme
    assert f"{val['omega_mean']:.3f}" in readme
    assert f"{val['final_policy_change']:.2e}" in readme


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
