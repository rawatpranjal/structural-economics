"""Faithfulness tests for the cake-eating tutorial.

These tests pin the three findings in
``bullshit-detector_cake-eating_2026-05-20.md``:

- Finding 1 (FALSE): the README claimed "choosing k=1 recovers VFI exactly".
  The MPI loop evaluates ``k`` policy sweeps starting from ``T V_n`` (the
  result of one Bellman improvement), so it is ``k=0`` inner sweeps, not
  ``k=1``, that reproduces a plain VFI update. A small standalone
  reimplementation of the cake-eating building blocks verifies this
  numerically; the README prose is checked for the corrected claim.
- Finding 2 (DATA DRIFT): the MPI pseudocode initialised ``V_eval <- V_n``,
  while the code initialises ``v_eval = v_imp`` with ``v_imp = T V_n``. The
  pseudocode must initialise the evaluation phase from ``T V_n``.
- Finding 3 (DATA DRIFT): the Equations section and the Takeaway stated the
  policy contraction is applied ``k`` times per outer step, while the code
  (and the Convergence prose) apply it ``k+1`` times. All three sections
  must agree on ``k+1``.

The README is treated as the generated artifact; the run.py source is the
ground truth for the prose strings, since run.py is the single entrypoint
that regenerates the README.
"""
import re
from pathlib import Path

import numpy as np
import pytest

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
README = (TUTORIAL_DIR / "README.md").read_text()
RUN_PY = (TUTORIAL_DIR / "run.py").read_text()


# ---------------------------------------------------------------------------
# Standalone reimplementation of the cake-eating MPI building blocks.
#
# This mirrors the loop in run.py (bellman_step + k policy-evaluation sweeps,
# the evaluation phase initialised from T V_n) without executing the whole
# tutorial. It exists so Finding 1 can be checked numerically.
# ---------------------------------------------------------------------------
BETA = 0.9
N_GRID = 120
N_CONS = 200
W_MIN = 0.01
W_MAX = 1.0
W_GRID = np.linspace(W_MIN, W_MAX, N_GRID)


def _u(c):
    return np.log(np.maximum(c, 1e-15))


def _analytical_v(w):
    return (
        np.log((1 - BETA) * np.maximum(w, 1e-15)) / (1 - BETA)
        + BETA * np.log(BETA) / (1 - BETA) ** 2
    )


def _v_interp(wprime, v):
    result = np.interp(wprime, W_GRID, v)
    below = wprime < W_GRID[0]
    if np.any(below):
        result[below] = _analytical_v(wprime[below])
    return result


def _bellman_step(v):
    """One application of T: argmax over consumption at each grid point."""
    v_new = np.zeros(N_GRID)
    policy_c = np.zeros(N_GRID)
    for ia in range(N_GRID):
        cake = W_GRID[ia]
        c_grid = np.linspace(1e-8, cake * 0.9999, N_CONS)
        wprime = cake - c_grid
        values = _u(c_grid) + BETA * _v_interp(wprime, v)
        best = int(np.argmax(values))
        v_new[ia] = values[best]
        policy_c[ia] = c_grid[best]
    return v_new, policy_c


def _policy_eval_sweep(v, policy_c):
    """One Bellman application under a fixed policy: no inner maximization."""
    wprime = W_GRID - policy_c
    return _u(policy_c) + BETA * _v_interp(wprime, v)


def _mpi_outer_step(v, k_inner):
    """One MPI outer step, mirroring run.py lines 136-142.

    The evaluation phase initialises from ``v_imp = T v`` and then applies
    ``k_inner`` policy-evaluation sweeps.
    """
    v_imp, policy = _bellman_step(v)
    v_eval = v_imp
    for _ in range(k_inner):
        v_eval = _policy_eval_sweep(v_eval, policy)
    return v_eval


# ---------------------------------------------------------------------------
# Finding 1: k=0 (not k=1) recovers a VFI update in this implementation.
# ---------------------------------------------------------------------------
def test_finding1_violated_invariant_k1_does_not_equal_vfi():
    """Violated invariant: the buggy README claimed k=1 == VFI.

    On the implementation as written, k=1 applies T_pi once on top of the
    Bellman update, so it does NOT match a plain VFI update. This test
    documents the bug: it passes because the two genuinely differ.
    """
    v0 = _u(W_GRID)
    v_vfi, _ = _bellman_step(v0)
    v_mpi_k1 = _mpi_outer_step(v0, k_inner=1)
    assert not np.allclose(v_vfi, v_mpi_k1, rtol=1e-10, atol=1e-12)


def test_finding1_honest_fix_k0_recovers_vfi():
    """Honest-fix pass condition: k=0 inner sweeps reproduces a VFI update.

    With zero evaluation sweeps the MPI step leaves v_eval = v_imp = T V_n,
    which is exactly the VFI update.
    """
    v0 = _u(W_GRID)
    v_vfi, _ = _bellman_step(v0)
    v_mpi_k0 = _mpi_outer_step(v0, k_inner=0)
    assert v_mpi_k0 == pytest.approx(v_vfi, rel=1e-10, abs=1e-12)


def test_finding1_readme_claims_k0_not_k1():
    """Honest-fix pass condition: README states k=0 recovers VFI, not k=1."""
    # The corrected claim must mention k=0 recovering VFI.
    assert re.search(r"k\s*=\s*0[^.]*value function iteration", README)
    # The false "k=1 recovers VFI" claim must be gone.
    assert not re.search(r"k\s*=\s*1[^.]*recovers value function iteration", README)
    assert not re.search(r"k\s*=\s*1[^.]*makes MPI identical to VFI", README)


# ---------------------------------------------------------------------------
# Finding 2: MPI pseudocode must initialise V_eval from T V_n, not V_n.
# ---------------------------------------------------------------------------
def _mpi_pseudocode_block():
    """Extract the MPI pseudocode 'V_eval <- ...' initialisation line."""
    m = re.search(r"V_eval\s*<-\s*(.+)", RUN_PY)
    return m.group(1).strip() if m else None


def test_finding2_honest_fix_pseudocode_inits_from_t_vn():
    """Honest-fix pass condition: pseudocode initialises V_eval from T V_n."""
    init = _mpi_pseudocode_block()
    assert init is not None, "MPI pseudocode 'V_eval <- ...' line not found"
    # The init must reference the Bellman update T V_n, matching v_eval = v_imp.
    assert "T V_n" in init


def test_finding2_pseudocode_not_initialised_from_vn():
    """The stale 'V_eval <- V_n' initialisation must be gone."""
    assert not re.search(r"V_eval\s*<-\s*V_n\b", RUN_PY)


# ---------------------------------------------------------------------------
# Finding 3: Equations and Takeaway must say the contraction is applied
# k+1 times, agreeing with the code and the Convergence prose.
# ---------------------------------------------------------------------------
def _section(name, text):
    """Return the body of a '### name' or '## name' README section."""
    m = re.search(
        rf"#+\s*{re.escape(name)}\b(.*?)(?=\n#+\s|\Z)", text, re.S
    )
    return m.group(1) if m else ""


def test_finding3_equations_mpi_uses_k_plus_1():
    """Honest-fix: the MPI iterate in Equations composes T_pi k+1 times."""
    equations = _section("Method 2: Modified Policy Iteration", README)
    assert equations, "MPI Equations subsection not found"
    # The MPI iterate must show a k+1 exponent on the policy contraction.
    assert re.search(r"T_\{?\\pi[^}]*\}?\^?\{?[^}]*k\s*\+\s*1", equations) or (
        "k+1" in equations or "k + 1" in equations
    )


def test_finding3_takeaway_uses_k_plus_1():
    """Honest-fix: the Takeaway states the contraction runs k+1 times."""
    takeaway = _section("Takeaway", README)
    assert takeaway, "Takeaway section not found"
    assert "k+1" in takeaway or "k + 1" in takeaway
    # The stale "a total of k times" must be gone from the Takeaway.
    assert not re.search(r"total of\s*\$?k\$?\s*times", takeaway)
