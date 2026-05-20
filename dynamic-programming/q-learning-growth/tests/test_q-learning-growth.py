"""Faithfulness tests for the q-learning-growth tutorial.

Findings from bullshit-detector_q-learning-growth_2026-05-20.md:
  1. policy MAE reported without disclosing that boundary rows are excluded.
  2. comparison-table "samples" column conflates VFI's deterministic
     sweep-evaluations with stochastic sampled transitions.

Each finding gets a violated-invariant test (holds on the buggy state) and an
honest-fix test (holds only after the repair). run.py runs the whole tutorial
on import, so claims are checked against run.py / README.md source text and
the regenerated comparison CSV.
"""
import csv
from pathlib import Path

FOLDER = Path(__file__).resolve().parent.parent
RUN_PY = (FOLDER / "run.py").read_text()
README = (FOLDER / "README.md").read_text().lower()


def _comparison_columns():
    with open(FOLDER / "tables" / "algorithm-comparison.csv") as fh:
        return next(csv.reader(fh))


# --- Finding 1: interior-only MAE must be disclosed in the report ---

def test_finding1_violated_invariant():
    """Buggy state: run.py masks the 3 lowest and 3 highest capital rows out
    of the MAE (interior_mask[:3] / [-3:] = False). The mask is real."""
    assert "interior_mask[:3] = False" in RUN_PY
    assert "interior_mask[-3:] = False" in RUN_PY


def test_finding1_honest_fix():
    """Honest state: the README discloses the boundary exclusion in prose or
    a table caption, so the reader knows the MAE numbers are interior-only."""
    assert "interior" in README or "boundary" in README


# --- Finding 2: "samples" column mislabels VFI's deterministic evaluations ---

def test_finding2_violated_invariant():
    """Buggy state: a bare column literally named 'samples' implies stochastic
    draws for every row, including value iteration. FAILS once renamed."""
    assert "samples" in _comparison_columns()


def test_finding2_honest_fix():
    """Honest state: the count column carries a name that does not imply
    stochastic sampling, and an explicit evaluation-type column distinguishes
    deterministic sweeps from stochastic samples."""
    cols = _comparison_columns()
    assert "samples" not in cols
    assert "state-action evaluations" in cols
    assert "evaluation type" in cols
    with open(FOLDER / "tables" / "algorithm-comparison.csv") as fh:
        rows = list(csv.DictReader(fh))
    vfi = next(r for r in rows if r["algorithm"] == "value iteration")
    assert vfi["evaluation type"] == "deterministic sweeps"
