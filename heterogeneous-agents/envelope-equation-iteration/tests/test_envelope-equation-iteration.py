"""Faithfulness tests for the envelope-equation-iteration tutorial.

Findings from bullshit-detector_envelope-equation-iteration_2026-05-20.md:
  F1 MISLABELED: title says "Persistent Income"; code implements IID income.
  F2 DILUTED:    convergence figure compares value-level vs consumption-level
                 error without disclosure.
  F3 DATA DRIFT: MPC printed at .3f in prose vs .4f in table.
  F4 DATA DRIFT: mean assets printed at .2f in prose vs .4f in table.
  F5 DILUTED:    Takeaway "EGP is faster here" contradicts the figure's
                 "not a timing claim" disclaimer.

run.py runs the whole tutorial on import, so claims are tested against the
README.md text and the run.py source string.
"""

from pathlib import Path

TUT = Path(__file__).resolve().parents[1]
README = (TUT / "README.md").read_text()
RUN_PY = (TUT / "run.py").read_text()


# --- Finding 1: title mislabels an IID-income model as "Persistent Income" ---

def test_f1_violated_invariant_title_says_persistent():
    """Buggy: the README title contains the word 'Persistent'."""
    assert "Persistent" in README.splitlines()[0]


def test_f1_honest_fix_title_names_iid():
    """Fixed: the title no longer says 'Persistent' and names IID income."""
    title = README.splitlines()[0]
    assert "Persistent" not in title
    assert "IID" in title


# --- Finding 2: convergence figure mixes value-level and consumption-level error ---

def test_f2_violated_invariant_no_metric_disclosure():
    """Buggy: the convergence figure prose does not disclose the metric mismatch."""
    # The buggy description said "needs more iterations" with no value-level note.
    assert "value-level error" not in README


def test_f2_honest_fix_metric_difference_disclosed():
    """Fixed: the convergence prose discloses VFI tracks a value-level error."""
    assert "value-level error" in README


# --- Finding 3: MPC precision drift between prose and table ---

def test_f3_violated_invariant_mpc_precision_drift():
    """Buggy: both '0.220' (prose) and '0.2197' (table) appear in the README."""
    assert "0.220" in README and "0.2197" in README


def test_f3_honest_fix_mpc_single_precision():
    """Fixed: a single precision is used for the MPC throughout the README."""
    assert not ("0.220" in README and "0.2197" in README)


# --- Finding 4: mean-assets precision drift between prose and table ---

def test_f4_violated_invariant_mean_assets_precision_drift():
    """Buggy: prose says 'Mean assets are 0.41.' while the table says '0.4124'."""
    assert "Mean assets are 0.41." in README and "0.4124" in README


def test_f4_honest_fix_mean_assets_single_precision():
    """Fixed: the prose mean-assets value matches the 4-decimal table value."""
    assert "Mean assets are 0.41." not in README
    assert "Mean assets are 0.4124" in README


# --- Finding 5: Takeaway timing claim contradicts the figure disclaimer ---

def test_f5_violated_invariant_timing_claim_contradiction():
    """Buggy: README both says 'not a timing claim' and 'EGP is faster here'."""
    assert "not a timing claim" in README and "EGP is faster here" in README


def test_f5_honest_fix_no_timing_contradiction():
    """Fixed: the bare 'EGP is faster here' timing claim is gone."""
    assert "EGP is faster here" not in README
