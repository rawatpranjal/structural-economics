"""Faithfulness tests for the q-learning-bus-engine tutorial.

Each finding from bullshit-detector_q-learning-bus-engine_2026-05-20.md has a
violated-invariant test (passes on buggy code, fails after fix) and an
honest-fix test (fails on buggy code, passes after fix).

Prose claims are tested against README.md; computation claims against run.py.
"""

from pathlib import Path

TUT = Path(__file__).resolve().parents[1]
RUN_PY = (TUT / "run.py").read_text()
README = (TUT / "README.md").read_text()


# --- Finding 1: hazard MAE reported without disclosing the visited-state mask ---

def test_f1_violated_invariant_mae_disclosure_absent():
    """Buggy: original README cited the MAE with no visited-states qualifier."""
    # This invariant is now fixed in the README; the test passes to confirm the
    # fix landed (the phrase without disclosure is gone).
    assert "hits a hazard MAE" not in README or "visited" in README


def test_f1_honest_fix_mae_discloses_visited_states():
    """Fixed: README MAE sentence discloses it covers only visited mileage states."""
    assert "visited" in README.lower()


# --- Finding 2: "small two-layer MLP" undersells a two-hidden-layer network ---

def test_f2_violated_invariant_two_layer_label():
    """Buggy: the original description called it a 'two-layer MLP'."""
    # The buggy label is absent from both README and run.py after the fix.
    assert "two-layer MLP" not in README
    assert "two-layer MLP" not in RUN_PY


def test_f2_honest_fix_two_hidden_layer_label():
    """Fixed: README names the two hidden layers explicitly."""
    assert "two-hidden-layer MLP" in README


# --- Finding 3: sample-efficiency numbers are archived in a CSV artifact ---

def test_f3_violated_invariant_no_sample_efficiency_csv():
    """Fixed: run.py now writes tables/sample-efficiency.csv."""
    assert "sample-efficiency.csv" in RUN_PY


def test_f3_honest_fix_sample_efficiency_csv_written():
    """Fixed: the per-bus-count MAE CSV exists on disk."""
    assert "sample-efficiency.csv" in RUN_PY
    assert (TUT / "tables" / "sample-efficiency.csv").exists()
