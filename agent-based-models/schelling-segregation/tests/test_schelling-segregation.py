"""Faithfulness tests for the Schelling segregation tutorial.

Covers the bullshit-detector finding (2026-05-20):
  Finding 1: the threshold-sweep table renders tau = 1/3 as the 3-decimal
  label "0.333". The computation uses the exact rational 1.0/3.0; a reader
  reading the CSV/README label back as a float gets 0.333, which differs
  from 1/3 by ~3.3e-4. The displayed label must round-trip to the float
  actually used in the sweep.

run.py executes the full model on import, so these tests read the committed
CSV artifact (tables/threshold-sweep.csv) rather than importing run.py.
"""
from pathlib import Path

import pandas as pd
import pytest

TUT = Path(__file__).resolve().parents[1]
CSV = TUT / "tables" / "threshold-sweep.csv"


def tau_column() -> pd.Series:
    return pd.read_csv(CSV)["Threshold tau"]


def one_third_row_index() -> int:
    """Row whose tau is nearest 1/3 in the committed CSV."""
    taus = tau_column().astype(float)
    return int((taus - 1.0 / 3.0).abs().idxmin())


# --- Finding 1: tau = 1/3 label precision -------------------------------------

def test_finding1_violated_invariant_tau_label_rounded():
    """Violated invariant: the 1/3 row label does NOT round-trip to 1.0/3.0.

    PASSES on the buggy CSV (label "0.333" differs from 1/3 by ~3.3e-4);
    FAILS once the honest fix stores the label at full precision.
    """
    idx = one_third_row_index()
    label = float(tau_column().iloc[idx])
    assert abs(label - 1.0 / 3.0) > 1e-6


def test_finding1_honest_fix_tau_label_exact():
    """Honest fix: the 1/3 row label round-trips to the float 1.0/3.0.

    FAILS on the buggy CSV; PASSES once the label is stored at full precision.
    """
    idx = one_third_row_index()
    label = float(tau_column().iloc[idx])
    assert abs(label - 1.0 / 3.0) < 1e-10


def test_other_tau_labels_still_roundtrip():
    """Every tau label must round-trip to the float used in the sweep.

    The exact tau_grid used by run.py; each committed label, read back as a
    float, must match its grid value. Guards against the fix degrading the
    precision of the non-1/3 rows.
    """
    grid = [0.20, 0.225, 0.25, 0.275, 0.30, 1.0 / 3.0, 0.35,
            0.375, 0.40, 0.425, 0.45, 0.475, 0.50]
    labels = [float(v) for v in tau_column()]
    assert len(labels) == len(grid)
    for label, expected in zip(sorted(labels), sorted(grid)):
        assert abs(label - expected) < 1e-10


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
