"""Faithfulness test for the sequential-search-ursu tutorial.

Covers the single non-HOLDS finding from
bullshit-detector_sequential-search-ursu_2026-05-20.md:

  F1 (DILUTED, HIGH) -- the Results prose claims the inside purchase share
      falls as search costs rise, but every row of the committed
      search-cost-counterfactual.csv reports an inside purchase share of
      1.0. The outside-option mechanism is coded correctly; the calibration
      simply leaves no consumer unable to find a positive match.

The code and the CSV agree: the inside share is genuinely flat at 1.0. The
faithful fix is therefore a prose fix -- the Results text must describe what
the table shows (search depth falls, inside share stays near one), not a
falling inside-demand curve that the table never exhibits.

Violated-invariant test: passes while the prose still claims the inside
share falls. Honest-fix test: passes once the prose matches the table.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TUTORIAL = HERE.parent
RUN_PY = (TUTORIAL / "run.py").read_text()
README = (TUTORIAL / "README.md").read_text()
CF = pd.read_csv(TUTORIAL / "tables" / "search-cost-counterfactual.csv")


def test_committed_table_has_flat_inside_share():
    """Ground truth: the committed counterfactual table reports an inside
    purchase share of 1.0 in every row, so any prose claiming a fall is the
    bug, not the data."""
    assert all(row["Inside purchase share"] == 1.0 for _, row in CF.iterrows())


def test_average_searches_does_fall():
    """The genuine, table-supported counterfactual effect: average searches
    falls monotonically as the search-cost multiplier rises."""
    ordered = CF.sort_values("Search cost multiplier")
    assert list(ordered["Average searches"]) == sorted(
        ordered["Average searches"], reverse=True
    )


def test_f1_violated_invariant_prose_claims_inside_share_falls():
    """Violated invariant: passes while the buggy Results prose claims the
    inside purchase share falls; must FAIL after the prose is corrected."""
    assert "The inside purchase share falls" in RUN_PY


def test_f1_honest_fix_prose_matches_flat_inside_share():
    """Honest fix: the Results prose no longer claims a falling inside share
    and instead states it stays near one while search depth falls."""
    assert "The inside purchase share falls" not in RUN_PY
    assert "The inside purchase share falls" not in README
    # The corrected prose describes the real effect.
    lowered = README.lower()
    assert "inside purchase share" in lowered
    assert "near one" in lowered or "stays at one" in lowered or "remains at one" in lowered
