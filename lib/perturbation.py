"""First-order perturbation by Klein-style generalized Schur (QZ) decomposition.

Given a linear rational-expectations model in the canonical form

    A · E_t s_{t+1}  =  B · s_t

where s_t = (x_t, y_t) stacks n_x predetermined variables on top of n_y
forward-looking variables, this module returns the policy function

    y_t      = P · x_t
    x_{t+1}  = F · x_t

following Klein (2000), "Using the generalized Schur form to solve a
multivariate linear rational expectations model," J. Econ. Dynamics and
Control, 24(10), 1405–1423. This is the same algorithm Dynare uses for
``stoch_simul, order=1`` (see Villemot 2011, "Solving Rational Expectations
Models at First Order: What Dynare Does", CEPREMAP Working Paper 2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import ordqz


@dataclass
class KleinSolution:
    """Decision rule and state transition from Klein-style QZ."""

    F: np.ndarray
    P: np.ndarray
    eigenvalues: np.ndarray
    n_stable: int
    n_predetermined: int
    blanchard_kahn_satisfied: bool
    bk_message: str


def solve_klein(A: np.ndarray, B: np.ndarray, n_predetermined: int) -> KleinSolution:
    """Solve A E_t s_{t+1} = B s_t by ordered generalized Schur (QZ).

    Conventions follow Klein (2000) and scipy's ``ordqz``. The variable
    ordering must place predetermined (state) variables in the top
    ``n_predetermined`` rows of s; forward-looking variables fill the
    remaining rows.

    Args:
        A, B: square matrices of equal size; A multiplies E_t s_{t+1},
            B multiplies s_t.
        n_predetermined: number of predetermined variables (top of s).

    Returns:
        KleinSolution with policy matrices F (state transition) and P
        (decision rule), the sorted eigenvalues, stable-eigenvalue count,
        and a Blanchard–Kahn diagnostic.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n) or B.shape != (n, n):
        raise ValueError(f"A and B must be square and same size; got {A.shape}, {B.shape}")
    n_x = int(n_predetermined)
    n_y = n - n_x
    if not 0 <= n_x <= n:
        raise ValueError(f"n_predetermined must be in [0, {n}]; got {n_x}")

    # We want eigenvalues of A^{-1} B (stable means |lambda| < 1 for the
    # forward recursion s_{t+1} = A^{-1} B s_t). scipy's ordqz(X, Y) returns
    # generalized eigenvalues alpha/beta of det(X - lambda Y) = 0, i.e.,
    # eigenvalues of Y^{-1} X. Calling ordqz(B, A) therefore gives exactly
    # the eigenvalues we want, and sort='iuc' puts |lambda|<1 upper-left.
    BB, AA, alpha, beta, _, Z = ordqz(B, A, sort="iuc", output="complex")
    with np.errstate(divide="ignore", invalid="ignore"):
        eigvals = np.where(np.abs(beta) > 0, alpha / beta, np.full_like(alpha, np.inf))
    n_stable = int(np.sum(np.abs(eigvals) < 1.0))

    bk_ok = n_stable == n_x
    if n_stable < n_x:
        msg = f"indeterminacy: only {n_stable} stable eigenvalues for {n_x} predetermined vars"
    elif n_stable > n_x:
        msg = f"no solution: {n_stable} stable eigenvalues exceed {n_x} predetermined vars"
    else:
        msg = "Blanchard–Kahn satisfied"

    Z11 = Z[:n_x, :n_x]
    Z21 = Z[n_x:, :n_x]
    AA11 = AA[:n_x, :n_x]
    BB11 = BB[:n_x, :n_x]

    if n_x > 0 and np.linalg.cond(Z11) > 1e14:
        raise np.linalg.LinAlgError("Z11 is ill-conditioned; predetermined block is singular")

    if n_x == 0:
        F = np.zeros((0, 0))
    else:
        F = np.real(Z11 @ np.linalg.solve(AA11, BB11 @ np.linalg.inv(Z11)))

    if n_y == 0:
        P = np.zeros((0, n_x))
    else:
        P = np.real(Z21 @ np.linalg.inv(Z11)) if n_x > 0 else np.zeros((n_y, 0))

    return KleinSolution(
        F=F,
        P=P,
        eigenvalues=eigvals,
        n_stable=n_stable,
        n_predetermined=n_x,
        blanchard_kahn_satisfied=bk_ok,
        bk_message=msg,
    )
