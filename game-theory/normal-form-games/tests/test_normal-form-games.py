"""Faithfulness tests for the normal-form-games tutorial.

Covers bullshit-detector finding 1 (2026-05-20): the heat map and pseudocode
use the combined quantity max{d1, d2}, but the Equations section never defines
it. After the honest fix, the Equations section names the combined deviation
quantity.

The README is generated from run.py; both files are checked as source text.
"""

from pathlib import Path

_TUTORIAL = Path(__file__).resolve().parents[1]
_README = (_TUTORIAL / "README.md").read_text()
_RUN_PY = (_TUTORIAL / "run.py").read_text()


def test_violated_invariant_combined_quantity_undefined():
    """Buggy state: Equations never defines the combined quantity max{d1,d2}.

    Fails once the honest fix adds the combined-deviation definition.
    """
    has_combined_def = any(
        phrase in _README
        for phrase in [r"\max\lbrace d_1", r"\max\{d_1", "combined deviation"]
    )
    assert not has_combined_def


def test_honest_fix_combined_quantity_defined():
    """Honest fix: Equations names the combined deviation quantity max{d1,d2}.

    Fails on the current buggy README.
    """
    has_combined_def = any(
        phrase in _README
        for phrase in [r"\max\lbrace d_1", r"\max\{d_1", "combined deviation"]
    )
    assert has_combined_def


def test_pseudocode_still_uses_combined_quantity():
    """The pseudocode step that needs the definition stays present."""
    assert "max{d1(i,j), d2(i,j)}" in _RUN_PY
