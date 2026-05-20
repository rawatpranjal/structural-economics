"""Finding 1: README prose claims "about a dozen barrier values" but the code
list has 9 entries and the table reports "9 barrier values".

The barrier sequence is the single source of truth. The Results prose must
state the realized count (9), not a vague "dozen".
"""
from pathlib import Path

TUTORIAL = Path(__file__).resolve().parents[1]
README = TUTORIAL / "README.md"
RUN_PY = TUTORIAL / "run.py"


def _barrier_list_length() -> int:
    """Number of entries in the `barriers = [...]` list literal in run.py."""
    src = RUN_PY.read_text()
    line = next(ln for ln in src.splitlines() if ln.strip().startswith("barriers = ["))
    inside = line.split("[", 1)[1].rsplit("]", 1)[0]
    return len([tok for tok in inside.split(",") if tok.strip()])


def test_violated_invariant_prose_says_dozen_for_a_nine_element_list():
    """PASSES on buggy code: prose says "a dozen" while the list holds 9.

    After the honest fix this test FAILS, because "a dozen" is gone from
    the README.
    """
    assert _barrier_list_length() == 9
    assert "a dozen" in README.read_text()


def test_honest_fix_prose_states_nine_barrier_values():
    """FAILS on buggy code: README still says "dozen", not "9 barrier values".

    After the honest fix this test PASSES.
    """
    text = README.read_text()
    assert "9 barrier values" in text
    assert "dozen" not in text
