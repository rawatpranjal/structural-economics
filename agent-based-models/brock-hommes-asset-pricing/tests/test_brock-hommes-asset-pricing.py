"""Faithfulness tests for the Brock-Hommes asset pricing tutorial.

Covers the two bullshit-detector findings (2026-05-20):
  Finding 1: initial conditions x_lag/x0 not disclosed in Model Setup table.
  Finding 2: fundamentalist cost c_F = 0 not disclosed in Model Setup table.

The tutorial run.py executes the full model on import, so these tests inspect
the generated README.md text instead of importing run.py.
"""
from pathlib import Path

import pytest

TUT = Path(__file__).resolve().parents[1]
README = TUT / "README.md"


def readme_text() -> str:
    return README.read_text()


def model_setup_block(text: str) -> str:
    """The Model Setup section, between its header and the next section."""
    after = text.split("Model Setup", 1)[1]
    return after.split("Solution", 1)[0]


# --- Finding 1: initial conditions disclosure ---------------------------------

def test_finding1_violated_invariant_initial_conditions_absent():
    """Violated invariant: Model Setup omits the initial deviation values.

    PASSES on the buggy README (no initial-deviation row); FAILS once the
    honest fix adds the initial-condition rows to the Model Setup table.
    (Scoped to the Model Setup block: "0.10" also appears in a DOI string.)
    """
    block = model_setup_block(readme_text())
    assert "0.10" not in block and "0.12" not in block


def test_finding1_honest_fix_initial_conditions_disclosed():
    """Honest fix: the Model Setup table discloses an initial deviation value.

    FAILS on the buggy README; PASSES once the rows are added.
    """
    block = model_setup_block(readme_text())
    assert "0.10" in block and "0.12" in block


# --- Finding 2: fundamentalist cost disclosure --------------------------------

def test_finding2_violated_invariant_fundamentalist_cost_absent():
    """Violated invariant: README omits the fundamentalist cost c_F.

    PASSES on the buggy README; FAILS once a c_F row is added.
    """
    text = readme_text()
    assert "c_F" not in text and "Fundamentalist cost" not in text


def test_finding2_honest_fix_fundamentalist_cost_disclosed():
    """Honest fix: the Model Setup table discloses the fundamentalist cost.

    FAILS on the buggy README; PASSES once the c_F row is added.
    """
    text = readme_text()
    assert "c_F" in text or "Fundamentalist cost" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
