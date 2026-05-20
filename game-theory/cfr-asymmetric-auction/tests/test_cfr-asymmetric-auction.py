"""Faithfulness tests for the cfr-asymmetric-auction tutorial.

Each audited finding gets two tests:
  - violated_invariant: captures the bug; PASSES on the buggy state, FAILS after the fix.
  - honest_fix: captures the faithful state; FAILS on the buggy state, PASSES after the fix.

Prose claims are tested against README.md (the canonical hand-maintained file).
Computation/artifact tests read generated CSVs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
README = (HERE / "README.md").read_text()
EXPL = pd.read_csv(HERE / "tables" / "asymmetric-exploitability.csv")
SUMMARY = pd.read_csv(HERE / "tables" / "methods-summary.csv")


# --- Finding 1: exploitability decay rate claimed as O(1/sqrt(T)) ---

def test_finding1_violated_invariant():
    # The committed data decays at ~T^{-0.8}, not T^{-0.5}. The buggy state
    # is one where the README does NOT name the empirical O(T^{-0.8}) rate
    # in the Results section (only the theoretical 1/sqrt(T) bound appears).
    it = EXPL["Iteration"].values
    ex = EXPL["Exploitability"].values
    slope = np.polyfit(np.log(it[1:]), np.log(ex[1:]), 1)[0]
    assert slope < -0.6  # genuinely faster than -0.5
    # Buggy state: empirical rate not reported in README
    assert "O(T^{-0.8})" not in README


def test_finding1_honest_fix():
    # Honest fix: README names the empirical ~O(T^{-0.8}) decay.
    assert "O(T^{-0.8})" in README


# --- Finding 2: alpha = 3/2 claim has no committed artifact ---

def test_finding2_violated_invariant():
    # On the buggy artifacts, alpha is not recorded anywhere in the CSV.
    assert not any("alpha" in q.lower() for q in SUMMARY["Quantity"].values)


def test_finding2_honest_fix():
    # Honest fix: alpha_opt is written to the summary CSV and equals 3/2.
    row = SUMMARY.set_index("Quantity")
    key = [q for q in SUMMARY["Quantity"].values if "alpha" in q.lower()][0]
    assert abs(float(row.loc[key, "Value"]) - 1.5) < 1e-3
