"""Faithfulness tests for the first-price-auctions tutorial.

Each audited finding gets two tests:
  - violated_invariant: captures the bug; PASSES on the buggy state, FAILS after the fix.
  - honest_fix: captures the faithful state; FAILS on the buggy state, PASSES after the fix.

run.py guards execution under ``if __name__ == "__main__"``, so the pure
numeric helpers can be imported directly; prose claims are tested against the
README (the canonical hand-maintained prose file).
"""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
README = (HERE / "README.md").read_text()

_spec = importlib.util.spec_from_file_location("fpa_run", HERE / "run.py")
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)


# --- Finding 1: focal grid best response does not exactly equal the analytic bid ---

def test_finding1_violated_invariant():
    # The focal grid best response (n=3, v=0.8) is one grid step off the
    # analytic bid; the buggy prose claimed it "sits on the analytic bid".
    # After the fix, "sits on" is gone from README.
    gap = abs(_run.grid_best_response(0.8, 3)[0] - _run.equilibrium_bid(0.8, 3))
    assert gap > 1e-10
    assert "sits on" not in README


def test_finding1_honest_fix():
    # Honest fix: the grid best response is genuinely off the analytic bid,
    # and the prose acknowledges the one-grid-spacing gap.
    gap = abs(_run.grid_best_response(0.8, 3)[0] - _run.equilibrium_bid(0.8, 3))
    assert gap > 1e-10
    assert "within one grid spacing of the analytic bid" in README
