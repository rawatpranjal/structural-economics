"""Faithfulness tests for the projection-methods tutorial.

Audit: bullshit-detector_projection-methods_2026-05-20.md, Finding 1.
The table description said errors are evaluated on a dense grid "between
collocation nodes". The eval grid is linspace(lower, upper, 320), which spans
the full approximation interval, not the sub-intervals strictly between nodes.
"""

from pathlib import Path

RUN_PY = Path(__file__).resolve().parents[1] / "run.py"


def test_eval_grid_spans_full_interval():
    """Violated-invariant: passes because the eval grid spans the full interval.

    linspace(lower, upper, 320) is a true description of the code. It would
    fail only if the grid were restricted to interior node-to-node ranges.
    """
    src = RUN_PY.read_text()
    assert "np.linspace(lower, upper, 320)" in src


def test_table_description_not_between_nodes():
    """Honest-fix: passes once the table description drops 'between collocation nodes'.

    Fails on current buggy prose.
    """
    src = RUN_PY.read_text()
    assert "between collocation nodes" not in src
