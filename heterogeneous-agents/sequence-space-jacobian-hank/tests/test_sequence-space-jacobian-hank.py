"""Faithfulness tests for the sequence-space-jacobian-hank tutorial.

Each audited finding gets two tests:
  - violated_invariant: captures the bug; PASSES on the buggy state, FAILS after the fix.
  - honest_fix: captures the faithful state; FAILS on the buggy state, PASSES after the fix.

run.py executes the whole tutorial on import, so prose/claim tests read the
run.py source text and the generated artifacts instead of importing run.py.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RUN_PY = (HERE / "run.py").read_text()
DIAG = pd.read_csv(HERE / "tables" / "diagnostics.csv")


# --- Finding 1: anticipation-curve figure mislabeled as J^{C,r}_{0,s} columns ---

def test_finding1_violated_invariant():
    # Buggy prose calls the policy-space curves "columns of J^{C, r}_{0, s}".
    assert "columns of $J^{C, r}_{0, s}$" in RUN_PY


def test_finding1_honest_fix():
    # Honest fix: describe the plotted object as the date-0 policy
    # perturbation dc(a) averaged over skill, before distribution weighting.
    assert "columns of $J^{C, r}_{0, s}$" not in RUN_PY
    assert "skill-averaged date-0 policy perturbation" in RUN_PY


# --- Finding 2: complexity prose implies O(T|state|) total for the forward sweep ---

def test_finding2_violated_invariant():
    # The forward sweep restarts delta_D once per pulse date inside the
    # s-loop, so it is O(T^2 |state|); buggy prose never states that total.
    n_restarts = sum(
        1 for ln in RUN_PY.split("\n") if "delta_D = np.zeros_like" in ln
    )
    assert n_restarts == 1 and "O(T^2" not in RUN_PY


def test_finding2_honest_fix():
    # Honest fix: prose names the O(T^2 |state|) total cost of this sweep.
    assert "O(T^2" in RUN_PY


# --- Finding 3: Q1/Q5 consumption ratio not in any committed artifact ---

def test_finding3_violated_invariant():
    # On the buggy code, no quintile ratio row exists in the CSV.
    assert "Q1/Q5 peak consumption ratio" not in DIAG["Quantity"].values


def test_finding3_honest_fix():
    # Honest fix: the Q1/Q5 ratio is written to the diagnostics CSV.
    row = DIAG.set_index("Quantity").loc["Q1/Q5 peak consumption ratio", "Value"]
    assert abs(float(row) - 4.0) < 1.5


# --- Finding 4: HANK vs RA inflation comparison not in any committed artifact ---

def test_finding4_violated_invariant():
    assert "Peak HANK inflation response (annualized %)" not in DIAG["Quantity"].values


def test_finding4_honest_fix():
    df = DIAG.set_index("Quantity")
    hank = abs(float(df.loc["Peak HANK inflation response (annualized %)", "Value"]))
    ra = abs(float(df.loc["Peak RA NK inflation response (annualized %)", "Value"]))
    assert hank > ra
