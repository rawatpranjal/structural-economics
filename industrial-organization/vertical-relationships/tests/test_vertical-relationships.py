"""Faithfulness tests for vertical-relationships tutorial.

Audit: bullshit-detector_vertical-relationships_2026-05-20.md
Finding 1 (MISLABELED, MED): the Solution Method pseudocode framed step 2 as a
"For a candidate wholesale price w" search, but solve_contracts derives w_DM
from a single closed-form expression with no loop or optimizer.

run.py guards main() under __main__, so importing the module is side-effect
free.
"""
import importlib.util
import inspect
from pathlib import Path

RUN_PY = Path(__file__).resolve().parents[1] / "run.py"
README = Path(__file__).resolve().parents[1] / "README.md"


def _load_run():
    spec = importlib.util.spec_from_file_location("vr_run", RUN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_finding1_violated_invariant_solver_is_closed_form():
    """Violated-invariant: solve_contracts has no loop or optimizer over w; it
    uses the closed-form w_DM. Holds before and after the fix (the fix only
    corrects the pseudocode prose, not the code)."""
    src = inspect.getsource(_load_run().solve_contracts)
    assert "optimize" not in src
    assert "linspace" not in src
    assert "for w" not in src
    assert "(a / b - cr + cm) / 2" in src


def test_finding1_honest_fix_pseudocode_names_closed_form():
    """Honest-fix: the rendered Solution Method pseudocode must not present a
    'For a candidate wholesale price' search; it must name the closed-form
    solution. Fails on the buggy README, passes after regeneration."""
    text = README.read_text()
    assert "For a candidate wholesale price" not in text
    assert "closed-form" in text
