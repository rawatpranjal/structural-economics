"""Faithfulness tests for the assetNews tutorial.

Covers bullshit-detector finding 1 (MISLABELED): model.mod declared a
third-order perturbation solve while the tutorial and run.py use first-order
perturbation throughout. After the honest fix, model.mod must declare
order=1 so a reader running `dynare model.mod` gets the same approximation
order the tutorial documents.
"""

from pathlib import Path

MOD_PATH = Path(__file__).resolve().parents[1] / "model.mod"


def _mod_text() -> str:
    return MOD_PATH.read_text()


def test_violated_invariant_order3_present():
    """Violated-invariant test for finding 1.

    Passes on the buggy code (order=3 present in model.mod), proving the
    mismatch between the .mod spec and the first-order tutorial. After the
    honest fix it FAILS, proving the mismatch was removed.
    """
    assert "order=3" in _mod_text()


def test_honest_fix_order1_declared():
    """Honest-fix test for finding 1.

    Fails on the buggy code. After the fix model.mod declares order=1,
    matching the first-order perturbation the tutorial documents.
    """
    text = _mod_text()
    assert "order=1" in text and "order=3" not in text
