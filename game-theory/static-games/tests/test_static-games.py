"""Faithfulness tests for the static-games tutorial.

Covers bullshit-detector finding 1 (2026-05-20): the code best response clips
at zero (np.maximum(0.0, .)), but the Equations section shows only the bare
interior FOC BR_i(q_j) = (a-c-b q_j)/(2b). After the honest fix, the equation
displays the non-negativity constraint that the code enforces.

The README is generated from run.py; both files are checked as source text.
"""

import inspect
import sys
from pathlib import Path

_TUTORIAL = Path(__file__).resolve().parents[1]
_README = (_TUTORIAL / "README.md").read_text()

sys.path.insert(0, str(_TUTORIAL))
import run as static_games_run  # noqa: E402


def test_violated_invariant_code_clips_but_equation_does_not():
    """Buggy state: code clips at zero, README equation shows bare interior FOC.

    Fails once the honest fix adds the non-negativity to the README equation.
    """
    src = inspect.getsource(static_games_run.cournot_best_response)
    code_clips = "np.maximum" in src and "0.0" in src
    equation_bare = (
        r"BR_i(q_j)=\frac{a-c-bq_j}{2b}" in _README
        and r"\max" not in _README.split("best response")[1].split("$$")[1]
    )
    assert code_clips and equation_bare


def test_honest_fix_equation_shows_non_negativity():
    """Honest fix: README BR equation includes the non-negativity clip.

    Fails on the current buggy README.
    """
    br_block = _README.split("best response")[1].split("$$")[1]
    assert r"\max" in br_block and "0," in br_block


def test_code_best_response_still_clips_at_zero():
    """The code best response keeps the non-negativity floor."""
    src = inspect.getsource(static_games_run.cournot_best_response)
    assert "np.maximum" in src and "0.0" in src
