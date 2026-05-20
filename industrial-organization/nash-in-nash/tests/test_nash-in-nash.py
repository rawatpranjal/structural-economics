"""Faithfulness tests for the nash-in-nash tutorial.

From bullshit-detector_nash-in-nash_2026-05-20.md, Finding 1 (DILUTED, LOW):
the figure-2 description claimed "The exercise does not recompute demand for
each tau". The tau sweep calls `bilateral_bargaining(tau_value)` once per tau,
and `bilateral_bargaining` calls `demand(counterfactual)` for every
hospital-insurer pair, so `demand` is in fact invoked on every sweep iteration.
The load-bearing economic fact is that the networks do not change with tau, so
full-agreement and disagreement demand are constant across tau and only the
surplus split moves.

Violated-invariant test: passed on the buggy prose. Honest-fix test: failed on
the buggy prose. After the fix the violated invariant fails and the honest fix
passes.

Prose now lives in README.md; run.py contains only computation and figure code.
"""
import re
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
RUN_SRC = (TUTORIAL_DIR / "run.py").read_text()
README_SRC = (TUTORIAL_DIR / "README.md").read_text()


def test_code_does_recompute_demand_inside_the_tau_sweep():
    """Ground truth the finding rests on: `bilateral_bargaining` calls
    `demand(...)` per pair, and the tau sweep calls `bilateral_bargaining`
    once per tau, so demand is recomputed on every sweep iteration.

    Holds before and after the prose fix -- it documents the code behaviour
    the prose must not contradict.
    """
    sweep = re.search(
        r"for i, tau_value in enumerate\(tau_grid\):.*?bilateral_bargaining",
        RUN_SRC, re.DOTALL,
    )
    assert sweep is not None
    bargain = RUN_SRC.split("def bilateral_bargaining")[1].split("\n\n    def ")[0]
    assert "demand(counterfactual)" in bargain


def test_finding1_violated_invariant_prose_claims_no_recompute():
    """Violated invariant: the figure description carries the unqualified
    claim that the exercise does not recompute demand for each tau, which
    contradicts the code.

    PASSED on the buggy prose; FAILS after the fix.
    Prose now lives in README.md.
    """
    assert "does not " in README_SRC and "recompute demand for each" in README_SRC


def test_finding1_honest_fix_prose_says_demand_held_fixed():
    """Honest fix: the figure description states the load-bearing fact -- the
    networks do not change with tau, so demand is held fixed and only the
    surplus split moves -- without the contradicted "does not recompute" claim.

    FAILED on the buggy prose; PASSES after the fix.
    Prose now lives in README.md.
    """
    assert "recompute demand for each" not in README_SRC
    assert "networks are the same" in README_SRC and "surplus split changes" in README_SRC
