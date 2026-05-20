"""Faithfulness tests for the rbc-irreversible-investment tutorial.

Covers bullshit-detector finding 1 (MISLABELED): the Solution Method
pseudocode said the exact off-grid boundary K'=(1-delta)K is added only to
the irreversible candidate set A_irr. The code computes that boundary
unconditionally and lets it win for BOTH models (the boundary block in
solve_rbc is not guarded by `if constrained`). For the standard model the
boundary can only win when negative investment is not optimal, so the std
solution is more accurate, not corrupted.

The honest fix corrects the prose: the code behaviour is economically right,
so the pseudocode is updated to describe what actually runs (the boundary
is evaluated for both models). The pseudocode lives inside the
add_solution_method string in run.py; README.md is generated from it.

run.py only runs main() under `__main__`, so importing it is safe and these
tests inspect solve_rbc source plus the run.py pseudocode text.
"""

import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from run import solve_rbc  # noqa: E402

RUN_PY = HERE.parent / "run.py"
SRC = RUN_PY.read_text()
SOLVE_SRC = inspect.getsource(solve_rbc)


def _boundary_block() -> str:
    """Code between the ev_boundary computation and the is_binding write."""
    return SOLVE_SRC.split("ev_boundary")[1].split("is_binding = use_boundary")[0]


def test_violated_invariant_boundary_unguarded_and_pseudocode_says_irr_only():
    """Violated-invariant test for finding 1.

    Passes on the buggy code. The boundary block in solve_rbc is not wrapped
    in an `if constrained:` guard (so it runs for both models), while the
    README pseudocode claims the boundary is added only to A_irr. After the
    honest prose fix, the pseudocode no longer makes the A_irr-only claim,
    so this test FAILS.
    """
    boundary_unguarded = "if constrained" not in _boundary_block()
    pseudocode_says_irr_only = "boundary K'=(1-delta)K_m to A_irr" in SRC
    assert boundary_unguarded and pseudocode_says_irr_only


def test_honest_fix_pseudocode_matches_both_models():
    """Honest-fix test for finding 1.

    Fails on the buggy code. After the fix the pseudocode states that the
    exact off-grid boundary is evaluated for both candidate sets, matching
    the unconditional boundary computation the code actually runs.
    """
    # The boundary code stays unconditional (it is economically correct).
    boundary_unguarded = "if constrained" not in _boundary_block()
    # The pseudocode no longer claims the boundary belongs only to A_irr.
    no_irr_only_claim = "boundary K'=(1-delta)K_m to A_irr" not in SRC
    # The corrected pseudocode names both candidate sets.
    names_both = "to both A_std and A_irr" in SRC
    assert boundary_unguarded and no_irr_only_claim and names_both
