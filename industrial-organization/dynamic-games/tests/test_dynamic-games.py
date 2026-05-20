"""Faithfulness tests for the dynamic-games tutorial.

The bullshit-detector audit (2026-05-20) flagged three findings:

Finding 1 (DILUTED/MED): the README claims the output is a pure-strategy
MPE at every state, but select_equilibrium silently falls back to the
joint-payoff argmax when no pure NE exists -- a profile that is NOT a
Nash equilibrium. The fallback is never disclosed. A probe confirms no
state in this calibration lacks a pure NE, so the honest fix replaces
the silent fallback with an explicit assertion: the MPE claim becomes
self-enforcing and unconditional.

Finding 2 (DILUTED/MED): the Results prose says "Firm 1 waits at the top
rung" as a universal statement over all five q1=4 states, but the
evidence table has only one top-rung row, (4,4). The honest fix adds all
five top-rung rows so the claim is fully covered by the table.

Finding 3 (DATA DRIFT/LOW): the pseudocode names a damping weight alpha
but its value 0.35 appears nowhere in the README. The honest fix adds a
Model Setup row stating alpha=0.35.

select_equilibrium / solve_game are pure functions safe to import. The
README is read as text.
"""
import importlib
import inspect
import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parents[1]
README = FOLDER / "README.md"
sys.path.insert(0, str(FOLDER))
run = importlib.import_module("run")


def _readme() -> str:
    return README.read_text()


# --- Finding 1: undisclosed no-NE fallback --------------------------------

def test_violated_silent_fallback_present_and_undisclosed():
    """Violated invariant: select_equilibrium carries a silent no-NE
    fallback and the README never discloses it. PASSES on the buggy
    state, FAILS once the fallback is replaced by an explicit assertion
    (or disclosed in the README)."""
    src = inspect.getsource(run.select_equilibrium)
    readme = _readme().lower()
    has_silent_fallback = "if not equilibria" in src and "raise" not in src and "assert" not in src
    undisclosed = "fallback" not in readme and "no pure" not in readme
    assert has_silent_fallback and undisclosed


def test_fixed_mpe_claim_unconditional():
    """Honest fix: select_equilibrium no longer silently returns a
    non-equilibrium profile. Either the fallback is gone, or it is an
    explicit assert/raise that fails loudly, or the README discloses it.
    FAILS on the buggy state, PASSES after the fix."""
    src = inspect.getsource(run.select_equilibrium)
    readme = _readme().lower()
    silent_fallback = (
        "if not equilibria" in src and "raise" not in src and "assert" not in src
    )
    assert (not silent_fallback) or ("fallback" in readme) or ("no pure" in readme)


# --- Finding 2: "waits at top rung" broader than evidence -----------------

def _top_rung_states_in_readme() -> list[str]:
    text = _readme()
    return [tok for tok in ["(4,0)", "(4,1)", "(4,2)", "(4,3)", "(4,4)"]
            if tok in text]


def test_violated_top_rung_coverage_incomplete():
    """Violated invariant: the README claims Firm 1 waits at the top rung
    but the results table covers only one of the five q1=4 states.
    PASSES on the buggy state, FAILS once all five are in the table."""
    assert len(_top_rung_states_in_readme()) < 5


def test_fixed_all_top_rung_states_in_table():
    """Honest fix: all five top-rung states appear in the README, so the
    "waits at top rung" claim is fully covered by the evidence table.
    FAILS on the buggy state, PASSES after the fix."""
    assert len(_top_rung_states_in_readme()) == 5


def test_fixed_firm1_waits_at_every_top_rung_state():
    """The underlying claim itself: firm 1 waits (action 0) at every q1=4
    state. Backs the prose with a recomputation."""
    sol = run.solve_game()
    import numpy as np
    policy = np.asarray(sol["policy"])
    assert all(policy[4, q2, 0] == 0 for q2 in range(5))


# --- Finding 3: damping weight alpha=0.35 undisclosed ---------------------

def test_violated_damping_alpha_value_missing():
    """Violated invariant: the README names the damping weight alpha but
    never states its value 0.35. PASSES on the buggy state, FAILS once
    the value is disclosed."""
    text = _readme()
    names_alpha = "alpha" in text
    value_missing = not any("alpha" in line and "0.35" in line
                            for line in text.splitlines())
    assert names_alpha and value_missing


def test_fixed_damping_alpha_value_disclosed():
    """Honest fix: a single README line names alpha and states 0.35.
    FAILS on the buggy state, PASSES after the fix."""
    text = _readme()
    assert any("alpha" in line and "0.35" in line for line in text.splitlines())
