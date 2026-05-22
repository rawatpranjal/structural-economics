"""Upwind finite-difference operators on a 1D state grid.

Promoted from the inline upwind logic in
`optimal-control/hjb-growth/run.py` and
`heterogeneous-agents/huggett-incomplete-markets/run.py`. The helpers
build the sparse generator `A` used by the implicit upwind HJB iteration
and apply the Kuhn-Tucker clip at a lower state-constraint boundary.

Conventions match the dense tutorials this helper serves:

- 1D state grid `x` with uniform spacing `dx`.
- Drift `s(x)` is signed: positive drift means the state moves right,
  negative drift means it moves left.
- Generator `A` is a continuous-time Markov generator: zero row sums,
  non-positive diagonal, tridiagonal under one-state drift.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def upwind_generator(
    grid: np.ndarray,
    drift: np.ndarray,
    *,
    boundary: str = "reflect",
) -> sp.csc_matrix:
    """Build the sparse upwind generator A on a 1D state grid.

    At interior nodes, the sub-diagonal entry is minus the negative part
    of the drift divided by the spacing, the super-diagonal entry is the
    positive part of the drift divided by the spacing, and the diagonal
    is set so each row sums to zero.

    Args:
        grid: Strictly increasing 1D array of node positions with
            uniform spacing `dx`.
        drift: Signed drift at each node, same length as `grid`. Sign
            chooses forward (positive) or backward (negative) one-sided
            difference. A drift of zero gives a zero row.
        boundary: How to treat the two endpoint rows. `"reflect"` forces
            no outflow off the grid: positive drift at the right
            boundary and negative drift at the left boundary contribute
            nothing. `"absorb"` keeps the upwind branch even if it would
            point off the grid, leaving the row a non-zero sub-Markov
            row.

    Returns:
        Sparse `len(grid) x len(grid)` generator in CSC format.
    """
    n = len(grid)
    if drift.shape != (n,):
        raise ValueError(
            f"drift shape {drift.shape} does not match grid length {n}"
        )
    dx = float(grid[1] - grid[0])

    s_plus = np.maximum(drift, 0.0)
    s_minus = np.minimum(drift, 0.0)

    if boundary == "reflect":
        # No outflow off the right boundary: kill positive drift at i = n-1.
        s_plus_eff = s_plus.copy()
        s_plus_eff[n - 1] = 0.0
        # No outflow off the left boundary: kill negative drift at i = 0.
        s_minus_eff = s_minus.copy()
        s_minus_eff[0] = 0.0
    elif boundary == "absorb":
        s_plus_eff = s_plus
        s_minus_eff = s_minus
    else:
        raise ValueError(f"unknown boundary mode: {boundary!r}")

    super_diag = s_plus_eff[: n - 1] / dx
    sub_diag = -s_minus_eff[1:n] / dx
    main_diag = -(s_plus_eff / dx) + (s_minus_eff / dx)

    A = (
        sp.diags(main_diag, 0, shape=(n, n))
        + sp.diags(sub_diag, -1, shape=(n, n))
        + sp.diags(super_diag, 1, shape=(n, n))
    )
    return A.tocsc()


def kt_state_constraint_clip(
    policy: np.ndarray,
    drift_at_min: float,
    *,
    constrained_policy: float,
) -> np.ndarray:
    """Apply the Kuhn-Tucker clip at a lower state-constraint boundary.

    If the unconstrained drift at the lower boundary is negative, the
    policy would push the state through the floor. The constraint binds
    and the policy is overridden by the value that holds the state
    exactly at the floor (zero drift).

    Args:
        policy: 1D policy on the state grid; first entry corresponds to
            the lower boundary.
        drift_at_min: Unconstrained drift implied by the policy at the
            lower boundary.
        constrained_policy: Policy value that makes drift zero at the
            lower boundary. For a savings problem this is income plus
            the return on the floor asset.

    Returns:
        Policy array with the first entry possibly overridden.
    """
    out = policy.copy()
    if drift_at_min < 0.0:
        out[0] = constrained_policy
    return out


def stationary_distribution(
    A: sp.spmatrix,
    *,
    dx: float = 1.0,
    fix_row: int = 0,
) -> np.ndarray:
    """Solve A^T g = 0 with a normalisation row replacing one redundant equation.

    The transpose `A^T` of a generator has a zero right-singular vector
    that integrates to the stationary distribution. The system is rank
    deficient, so one row is replaced by a normalisation constraint and
    the result is rescaled to sum to one in measure `dx`.

    Args:
        A: Generator matrix (continuous-time Markov generator), assumed
            to have zero row sums.
        dx: Quadrature weight for the trapezoidal rule used to normalise
            the density. Defaults to 1 for finite chains.
        fix_row: Index of the row to replace with the normalisation.

    Returns:
        Probability vector summing to 1 in measure `dx`.

    Notes:
        This helper is consumed primarily by the KFE prelim (#2). The
        upwind-FD prelim (#1) constructs `A` and may demonstrate the
        forward-equation duality only as a sanity check.
    """
    # TODO(prelim-2): when the KFE tutorial lands, factor the multi-state
    # block-generator path here too. For now, single 1D state.
    n = A.shape[0]
    AT = A.T.tolil()
    rhs = np.zeros(n)
    AT[fix_row, :] = 0.0
    AT[fix_row, fix_row] = 1.0
    rhs[fix_row] = 1.0
    g = spla.spsolve(AT.tocsc(), rhs)
    g = np.maximum(g, 0.0)
    g = g / (g.sum() * dx)
    return g


def forward_difference(values: np.ndarray, dx: float) -> np.ndarray:
    """Forward difference on a uniform grid; last entry left at zero.

    Args:
        values: 1D array of node values.
        dx: Uniform spacing.

    Returns:
        Array of the same length as `values`; entry `i` is
        `(values[i+1] - values[i]) / dx` for `i < n - 1`, and zero at
        the last index (use a boundary forcing rule instead).
    """
    n = len(values)
    out = np.zeros(n)
    out[: n - 1] = (values[1:n] - values[: n - 1]) / dx
    return out


def backward_difference(values: np.ndarray, dx: float) -> np.ndarray:
    """Backward difference on a uniform grid; first entry left at zero.

    Args:
        values: 1D array of node values.
        dx: Uniform spacing.

    Returns:
        Array of the same length as `values`; entry `i` is
        `(values[i] - values[i-1]) / dx` for `i >= 1`, and zero at the
        first index (use a boundary forcing rule instead).
    """
    n = len(values)
    out = np.zeros(n)
    out[1:n] = (values[1:n] - values[: n - 1]) / dx
    return out


def central_difference(values: np.ndarray, dx: float) -> np.ndarray:
    """Central difference on a uniform grid; endpoints set to one-sided.

    Provided so the prelim can show what happens when central
    differences are used in a setting where upwinding is required.
    """
    n = len(values)
    out = np.zeros(n)
    out[1 : n - 1] = (values[2:n] - values[: n - 2]) / (2.0 * dx)
    out[0] = (values[1] - values[0]) / dx
    out[n - 1] = (values[n - 1] - values[n - 2]) / dx
    return out
