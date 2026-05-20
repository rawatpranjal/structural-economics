"""Faithfulness tests for the huggett-incomplete-markets tutorial.

Findings from bullshit-detector_huggett-incomplete-markets_2026-05-20.md:
  F1 DATA DRIFT: relative value gap printed at .2f (prose) vs .3f (table).
  F2 DATA DRIFT: r* printed at .4f (prose/figure) vs .5f (Equations/table).
  F3 DATA DRIFT: precautionary wedge printed at .4f (prose) vs .5f (table).
  F4 DILUTED:    mean wealth hard-displayed as 0.00000 while the same quantity
                 appears as a non-zero residual in another table row.

run.py runs the whole tutorial on import, so claims are tested against the
generated README.md and tables/equilibrium.csv text.
"""

from pathlib import Path

TUT = Path(__file__).resolve().parents[1]
README = (TUT / "README.md").read_text()
CSV = (TUT / "tables" / "equilibrium.csv").read_text()


# --- Finding 1: relative value gap precision drift ---

def test_f1_violated_invariant_value_gap_precision_drift():
    """Buggy: README shows both '0.16%' (2dp) and '0.155%' (3dp) for one gap."""
    assert "0.16%" in README and "0.155%" in README


def test_f1_honest_fix_value_gap_single_precision():
    """Fixed: a single precision is used for the relative value gap."""
    assert not ("0.16%" in README and "0.155%" in README)


# --- Finding 2: r* precision drift ---

def test_f2_violated_invariant_r_star_precision_drift():
    """Buggy: README shows both '0.0350' (4dp) and '0.03499' (5dp) for r*."""
    assert "0.0350" in README and "0.03499" in README


def test_f2_honest_fix_r_star_single_precision():
    """Fixed: r* uses one (5-decimal) precision; the truncated '0.0350' is gone."""
    assert "= 0.0350$" not in README
    assert README.count("0.03499") >= 2


# --- Finding 3: precautionary wedge precision drift ---

def test_f3_violated_invariant_wedge_precision_drift():
    """Buggy: README shows the truncated '0.0150}' (4dp) alongside '0.01501' (5dp)."""
    # '0.0150}' is the 4dp form inside an inline-math wedge expression; the
    # bare '0.0150' substring also sits inside '0.01501', so match the
    # math-delimited truncated form to avoid the substring collision.
    assert "0.0150}" in README and "0.01501" in README


def test_f3_honest_fix_wedge_single_precision():
    """Fixed: the wedge uses one (5-decimal) precision; '0.0150}' truncation is gone."""
    assert "0.0150}" not in README
    assert README.count("0.01501") >= 2


# --- Finding 4: mean wealth masked to 0.00000 while residual row is non-zero ---

def _row_value(name: str) -> str:
    for row in CSV.splitlines():
        if row.startswith(name + ","):
            return row.split(",", 1)[1].strip()
    raise AssertionError(f"row {name!r} not found in equilibrium.csv")


def test_f4_violated_invariant_mean_wealth_masked():
    """Buggy: the Mean wealth row shows 0.00000 while a residual row is non-zero."""
    # On the buggy code mean wealth is hard-set to 0.00000.
    assert _row_value("Mean wealth E[a]") == "0.00000"


def test_f4_honest_fix_mean_wealth_shows_residual():
    """Fixed: the Mean wealth row shows the same non-zero residual as the residual row."""
    mean_wealth = _row_value("Mean wealth E[a]")
    residual = _row_value("Bond-market residual abs(S(r*))")
    assert mean_wealth != "0.00000"
    assert mean_wealth == residual
