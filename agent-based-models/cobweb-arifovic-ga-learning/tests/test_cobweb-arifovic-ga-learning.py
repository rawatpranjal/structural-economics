"""Faithfulness tests for the cobweb Arifovic GA-learning tutorial.

Covers the bullshit-detector finding (2026-05-20):
  Finding 1: README election step (6) writes the acceptance threshold as
  pi_{i,t}, which the symbol table defines as "firm i's realized profit",
  but the code compares the child against the *tournament-selected parent's*
  profit at slot i (parent_profits[i] = profits[parent_indices[i]]). The
  notation is ambiguous relative to the actual mechanics.

run.py executes the full model on import, so code-behaviour tests inspect the
run.py source text via file reads, and prose tests inspect the generated
README.md.
"""
from pathlib import Path

import pytest

TUT = Path(__file__).resolve().parents[1]
README = TUT / "README.md"
RUN_PY = TUT / "run.py"


def readme_text() -> str:
    return README.read_text()


def run_py_text() -> str:
    return RUN_PY.read_text()


def reproduce_source() -> str:
    """Source of the next-generation builder (reproduce), by header text."""
    text = run_py_text()
    start = text.index("def reproduce(")
    rest = text[start:]
    end = rest.index("\ndef ")
    return rest[:end]


# --- Finding 1: election-step notation ----------------------------------------

def test_election_compares_against_tournament_winner_profit():
    """Code fact: the election filter scores children against parent_profits,
    i.e. the tournament-selected parent's profit, not firm i's own profit.

    This anchors the finding: it PASSES on the current (correct) code and
    documents what the README prose must describe.
    """
    src = reproduce_source()
    assert "parent_profits = profits[parent_indices]" in src
    assert "child_profit[0] >= parent_profits[i]" in src
    assert "child_profit[1] >= parent_profits[i + 1]" in src


def test_finding1_violated_invariant_election_notation_unclarified():
    """Violated invariant: README step (6) uses pi_{i,t} as the election
    threshold without clarifying it is the tournament-selected parent's
    profit, while the symbol table defines pi_{i,t} as firm i's own profit.

    PASSES on the buggy README (no clarification); FAILS once the prose
    names the threshold as the selected parent's profit.
    """
    text = readme_text()
    assert "tournament-selected parent" not in text
    assert "selected parent" not in text


def test_finding1_honest_fix_election_notation_clarified():
    """Honest fix: README clarifies the election threshold is the
    tournament-selected parent's profit at slot i.

    FAILS on the buggy README; PASSES once the clarification is added.
    """
    text = readme_text()
    assert "selected parent" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
