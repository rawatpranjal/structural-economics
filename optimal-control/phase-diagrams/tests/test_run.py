"""Faithfulness tests for the phase-diagrams tutorial.

These tests guard the bullshit-detector finding from
bullshit-detector_phase-diagrams_2026-05-20.md, Finding 1:

    README.md and run.py claim "Arrows explain motion" in the
    path-selection figure, but the figure code (the fig3 block)
    contains no quiver call.

Importing run.py does not execute the tutorial: main() is gated by
``if __name__ == "__main__"``. So the fig3 block is inspected via
``inspect.getsource(run.main)`` without running solve_ivp or matplotlib.
"""
import importlib.util
import inspect
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
RUN_PY = TUTORIAL_DIR / "run.py"


def _load_run_module():
    spec = importlib.util.spec_from_file_location("phase_diagrams_run", RUN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fig3_block(source: str) -> str:
    """Slice out the fig3 plotting block from main()'s source."""
    after_fig3 = source.split("fig3", 1)[1]
    return after_fig3.split("fig3.tight_layout", 1)[0]


def test_fig3_block_isolated():
    """The fig3 slice must be a non-empty, bounded region of main()."""
    module = _load_run_module()
    source = inspect.getsource(module.main)
    assert "fig3" in source
    assert "fig3.tight_layout" in source
    block = _fig3_block(source)
    assert len(block) > 0


# --- Finding 1 -------------------------------------------------------------

def test_finding1_violated_invariant__fig3_has_no_quiver():
    """Violated-invariant test (from the audit, verbatim logic).

    Encodes the bug: the fig3 block contains no quiver call.

    PASSES on the buggy code (proves the bug is real).
    FAILS once the honest fix adds a quiver call to fig3 -- which is the
    expected, desired transition for a TDD violated-invariant test.
    """
    module = _load_run_module()
    source = inspect.getsource(module.main)
    block = _fig3_block(source)
    assert "quiver" not in block


def test_finding1_honest_fix__fig3_draws_quiver():
    """Honest-fix pass condition.

    The path-selection figure description claims "Arrows explain motion".
    For that claim to be faithful, the fig3 plotting block must actually
    draw arrows -- i.e. contain a quiver call.

    FAILS on the buggy code (no quiver in fig3 block).
    PASSES once the quiver call is added to fig3.
    """
    module = _load_run_module()
    source = inspect.getsource(module.main)
    block = _fig3_block(source)
    assert "quiver" in block, (
        "fig3 description claims 'Arrows explain motion' but the fig3 "
        "block draws no quiver arrows."
    )


def test_finding1_arrow_claim_backed_by_arrows():
    """If the figure description promises arrows, the figure must draw them.

    Couples the prose claim to the plotting code: if either the README or
    the run.py description string says 'Arrows', the fig3 block must have a
    quiver call. This both proves the original bug (claim present, quiver
    absent) and stays green after the honest fix (claim present, quiver
    present).
    """
    module = _load_run_module()
    source = inspect.getsource(module.main)
    block = _fig3_block(source)
    claims_arrows = "Arrows" in block or "arrows" in block
    if claims_arrows:
        assert "quiver" in block, (
            "fig3 block mentions arrows in its description but draws none."
        )
