"""Faithfulness tests for the q-learning-bus-engine tutorial.

Each finding from bullshit-detector_q-learning-bus-engine_2026-05-20.md has a
violated-invariant test (passes on buggy code, fails after fix) and an
honest-fix test (fails on buggy code, passes after fix).

run.py executes the whole tutorial on import, so claims about run.py source
are tested against the file text, and README claims against README.md.
"""

from pathlib import Path

TUT = Path(__file__).resolve().parents[1]
RUN_PY = (TUT / "run.py").read_text()
README = (TUT / "README.md").read_text()


# --- Finding 1: hazard MAE 0.0105 reported without disclosing the visited-state mask ---

def test_f1_violated_invariant_mae_disclosure_absent():
    """Buggy: closing prose cites the MAE with no visited-states qualifier."""
    idx = RUN_PY.index("hits a hazard MAE")
    context = RUN_PY[idx:idx + 240].lower()
    assert "visited" not in context and "masked" not in context


def test_f1_honest_fix_mae_discloses_visited_states():
    """Fixed: the MAE sentence discloses it covers only visited mileage states."""
    idx = RUN_PY.index("hits a hazard MAE")
    context = RUN_PY[idx:idx + 240].lower()
    assert "visited" in context or "masked" in context


# --- Finding 2: "small two-layer MLP" undersells a two-hidden-layer network ---

def test_f2_violated_invariant_two_layer_label():
    """Buggy: the network is described as a 'two-layer MLP'."""
    assert "two-layer MLP" in RUN_PY


def test_f2_honest_fix_two_hidden_layer_label():
    """Fixed: the description names the two hidden layers explicitly."""
    assert "two-hidden-layer MLP" in RUN_PY
    assert "two-layer MLP" not in RUN_PY


# --- Finding 3: sample-efficiency numbers are not archived in a CSV artifact ---

def test_f3_violated_invariant_no_sample_efficiency_csv():
    """Buggy: run.py never writes tables/sample-efficiency.csv."""
    assert "sample-efficiency.csv" not in RUN_PY


def test_f3_honest_fix_sample_efficiency_csv_written():
    """Fixed: run.py writes the per-bus-count MAE to a committed CSV."""
    assert "sample-efficiency.csv" in RUN_PY
    assert (TUT / "tables" / "sample-efficiency.csv").exists()
