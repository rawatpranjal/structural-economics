"""Faithfulness tests for the interpolation tutorial.

Covers the four findings in bullshit-detector_interpolation_2026-05-20.md.

Each finding has two tests:
  - violated-invariant test: PASSES on the original buggy prose, FAILS once fixed.
  - honest-fix test: FAILS on the original buggy prose, PASSES once fixed.

The data tests read the committed tables/comparison.csv. The prose tests read
the generated README.md. Convergence-slope tests recompute the sup-norm sweep
from run.py's own helpers.
"""
import csv
import inspect
from pathlib import Path

import numpy as np

import run

HERE = Path(__file__).resolve().parent.parent
CSV = HERE / "tables" / "comparison.csv"
README = HERE / "README.md"


def _csv_column(column):
    with open(CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {r["Method"]: float(r[column]) for r in rows}


def _readme():
    return README.read_text()


# -----------------------------------------------------------------------------
# Finding 1: "Cubic is uniformly smallest" on the smooth target.
# Data: PCHIP sup-error (0.781) < cubic sup-error (1.09) on the smooth target.
# -----------------------------------------------------------------------------

def test_f1_violated_invariant_cubic_named_uniformly_smallest():
    """Violated-invariant: original prose claims cubic is uniformly smallest.

    PASSES on the buggy README, FAILS once the prose is corrected.
    """
    assert "Cubic is uniformly smallest" in _readme()


def test_f1_honest_fix_smooth_winner_is_pchip():
    """Honest-fix: the smooth-target sup-norm winner is PCHIP, not cubic.

    The README must not state cubic is uniformly smallest on the smooth target.
    FAILS on the buggy README, PASSES once the prose is corrected.
    """
    smooth_sup = _csv_column("Smooth sup-error")
    assert smooth_sup["PCHIP (shape-preserving)"] < smooth_sup["Cubic spline (natural)"]
    assert "Cubic is uniformly smallest" not in _readme()


# -----------------------------------------------------------------------------
# Finding 2: "Cubic spline is the lowest-error choice on the smooth target".
# Data: PCHIP beats cubic on smooth sup-norm AND smooth L2.
# -----------------------------------------------------------------------------

def test_f2_violated_invariant_cubic_named_smooth_winner():
    """Violated-invariant: original prose names cubic the smooth-target winner.

    PASSES on the buggy README, FAILS once the prose is corrected.
    """
    assert "Cubic spline is the lowest-error choice on the smooth target" in _readme()


def test_f2_honest_fix_pchip_named_smooth_winner():
    """Honest-fix: README names PCHIP as the smooth-target winner.

    FAILS on the buggy README, PASSES once the prose is corrected.
    """
    smooth_sup = _csv_column("Smooth sup-error")
    smooth_l2 = _csv_column("Smooth L2 error")
    assert smooth_sup["PCHIP (shape-preserving)"] < smooth_sup["Cubic spline (natural)"]
    assert smooth_l2["PCHIP (shape-preserving)"] < smooth_l2["Cubic spline (natural)"]
    text = _readme()
    # The winner string is derived from argmin over fits["smooth"]; the method
    # name carries its parenthetical, so the rendered sentence reads
    # "PCHIP (shape-preserving) is the lowest-error choice on the smooth target".
    assert "PCHIP" in text.split("is the lowest-error choice on the smooth target")[0].splitlines()[-1]
    assert "Cubic spline is the lowest-error choice on the smooth target" not in text


# -----------------------------------------------------------------------------
# Finding 3: mpc=0.1 omitted from the Model Setup parameter table.
# -----------------------------------------------------------------------------

def _model_setup_has_mpc():
    return any(
        "MPC" in line and "0.1" in line
        for line in _readme().splitlines()
    )


def test_f3_violated_invariant_mpc_absent_from_setup():
    """Violated-invariant: original Model Setup table omits the MPC row.

    PASSES on the buggy README, FAILS once the MPC row is added.
    """
    assert not _model_setup_has_mpc()


def test_f3_honest_fix_mpc_listed_in_setup():
    """Honest-fix: Model Setup lists MPC = 0.1.

    The kinked target's slope above the kink is (1+r)*mpc; a reader cannot
    reproduce it without this value. FAILS on the buggy README, PASSES once
    the MPC row is added.
    """
    default_mpc = inspect.signature(run.make_consumption_policy).parameters["mpc"].default
    assert default_mpc == 0.1
    assert _model_setup_has_mpc()


# -----------------------------------------------------------------------------
# Finding 4: "linear slope -2", "cubic and PCHIP drop at roughly slope -4".
# Recomputed log-log slopes on the smooth target are -1.5, -1.7, -2.0; the
# log singularity of V(W) at W -> 0 suppresses the asymptotic O(h^4) rate.
# -----------------------------------------------------------------------------

def _smooth_slopes():
    """Log-log sup-norm slope on the smooth target for each method."""
    beta = 0.9
    V = run.make_cake_value(beta=beta)
    node_counts = np.array([5, 10, 20, 40, 80])
    slopes = {}
    for name, method_fn, *_ in run.METHODS:
        errs = np.array([
            run.errors_at_nodes(V, method_fn, int(n), 0.05, 1.0)["sup_err"]
            for n in node_counts
        ])
        slopes[name] = float(np.polyfit(np.log(node_counts), np.log(errs), 1)[0])
    return slopes


def test_f4_violated_invariant_readme_claims_slope_minus_4():
    """Violated-invariant: original prose claims cubic and PCHIP slope -4.

    PASSES on the buggy README, FAILS once the prose reports the real slopes.
    """
    assert "drop at roughly slope $-4$" in _readme()


def test_f4_honest_fix_slopes_match_recomputed_values():
    """Honest-fix: the recomputed slopes are nowhere near -4.

    Cubic and PCHIP slopes on this target are between -1.5 and -2.5, far from
    -4: the log singularity of V(W) at the left edge kills the asymptotic
    O(h^4) cubic rate. The README must not claim slope -4.
    """
    slopes = _smooth_slopes()
    cubic = slopes["Cubic spline (natural)"]
    pchip = slopes["PCHIP (shape-preserving)"]
    assert -2.5 < cubic < -1.0
    assert -2.5 < pchip < -1.5
    assert "drop at roughly slope $-4$" not in _readme()


# -----------------------------------------------------------------------------
# Finding 5 (recheck 2026-05-20): Takeaway claims cubic gives the best
# convergence on smooth functions. The Results section reports steeper log-log
# slope for PCHIP (-2.0) than cubic (-1.7), so PCHIP, not cubic, converges
# fastest on the smooth target. The Takeaway prose contradicts Results.
# -----------------------------------------------------------------------------

def test_f5_violated_invariant_takeaway_names_cubic_best_convergence():
    """Violated-invariant: original Takeaway names cubic the best converger.

    PASSES on the buggy README, FAILS once the Takeaway is corrected.
    """
    assert "Natural cubic spline gives the best convergence on smooth" in _readme()


def test_f5_honest_fix_takeaway_convergence_winner_is_pchip():
    """Honest-fix: PCHIP has the steepest smooth-target slope, so the Takeaway
    must not credit cubic with the best convergence.

    The recomputed log-log slopes put PCHIP below cubic (steeper, faster
    convergence). FAILS on the buggy README, PASSES once the Takeaway is fixed.
    """
    slopes = _smooth_slopes()
    cubic = slopes["Cubic spline (natural)"]
    pchip = slopes["PCHIP (shape-preserving)"]
    assert pchip < cubic
    assert "Natural cubic spline gives the best convergence on smooth" not in _readme()
