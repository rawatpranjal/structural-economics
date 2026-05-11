"""JAX-native first-order perturbation by complex Schur with implicit-IFT JVP.

Mirrors ``lib.perturbation.solve_klein`` but is jax-traceable, so ``jax.grad``
flows through it. JAX has neither generalized Schur (QZ) nor a Schur-reordering
primitive, and ``jax.scipy.linalg.schur`` has no autodiff rule. We work around
all three:

1. Reduce the pencil ``A E_t s_{t+1} = B s_t`` (with ``A`` invertible) to the
   standard eigenproblem ``M = A^{-1} B`` via ``jnp.linalg.solve``.
2. Compute complex Schur ``M = Z T Z^H`` with
   ``jax.scipy.linalg.schur(M, output="complex")``.
3. Bubble-sort the diagonal of ``T`` ascending by ``|lambda|`` using
   adjacent-pair Givens rotations applied to both ``T`` and ``Z``. With
   Blanchard-Kahn satisfied this puts the stable eigenvalues in the top-left.
4. Extract the policy from the sorted invariant subspace:
   ``F = Z11 T11 Z11^{-1}``, ``P = Z21 Z11^{-1}``.
5. For gradients, register a ``jax.custom_jvp`` that solves the implicit
   Klein equations
       (A_xx + A_xy P) F = B_xx + B_xy P
       (A_yx + A_yy P) F = B_yx + B_yy P
   for tangents (dF, dP) given (dA, dB). Schur is only on the primal path;
   the JVP never differentiates through it.

Caveats: ``A`` must be invertible. The JVP rule requires Blanchard-Kahn at
the linearization point (otherwise the implicit equations don't hold at the
returned F, P). Callers must keep theta inside the determinacy region.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


@dataclass
class KleinSolutionJax:
    """Decision rule and state transition from JAX-side Schur reordering."""

    F: jnp.ndarray
    P: jnp.ndarray
    eigenvalues: jnp.ndarray
    n_stable: jnp.ndarray
    n_predetermined: int
    bk_satisfied: jnp.ndarray


def _swap_step(T: jnp.ndarray, Z: jnp.ndarray, i: int):
    """Conditionally swap diagonal entries (i, i+1) of upper-triangular T.

    Applies a 2x2 unitary Givens rotation that maps the right eigenvector of
    ``T[i+1, i+1]`` in the (i, i+1) block to ``e_1``, then rotates rows
    (i, i+1) of T by G, columns (i, i+1) of T by G^H, and columns (i, i+1)
    of Z by G^H. The swap is gated on ``|T[i, i]| > |T[i+1, i+1]|`` so the
    pass-by-pass action sorts the diagonal in ascending ``|lambda|``.
    """
    a = T[i, i]
    b = T[i, i + 1]
    d = T[i + 1, i + 1]

    should_swap = jnp.abs(a) > jnp.abs(d)

    diff = d - a
    r2 = jnp.abs(b) ** 2 + jnp.abs(diff) ** 2
    r2_safe = jnp.where(r2 > 1e-30, r2, jnp.asarray(1.0, dtype=r2.dtype))
    r = jnp.sqrt(r2_safe)
    c = jnp.conj(b) / r
    s = jnp.conj(diff) / r
    degenerate = r2 < 1e-30
    c = jnp.where(degenerate, jnp.asarray(1.0, dtype=T.dtype), c)
    s = jnp.where(degenerate, jnp.asarray(0.0, dtype=T.dtype), s)

    row0 = jnp.stack([c, s])
    row1 = jnp.stack([-jnp.conj(s), jnp.conj(c)])
    G = jnp.stack([row0, row1])
    GH = jnp.conj(G).T

    rows = T[i:i + 2, :]
    T_new = T.at[i:i + 2, :].set(G @ rows)
    cols = T_new[:, i:i + 2]
    T_new = T_new.at[:, i:i + 2].set(cols @ GH)

    Z_cols = Z[:, i:i + 2]
    Z_new = Z.at[:, i:i + 2].set(Z_cols @ GH)

    T_out = jnp.where(should_swap, T_new, T)
    Z_out = jnp.where(should_swap, Z_new, Z)
    return T_out, Z_out


def _reorder_schur_ascending(T: jnp.ndarray, Z: jnp.ndarray):
    """Bubble-sort the diagonal of (T, Z) ascending by ``|lambda|``.

    ``n - 1`` passes of length ``n - 1`` adjacent-swap steps suffice to sort
    any permutation. Loops are unrolled in Python so the trace is static.
    """
    n = T.shape[0]
    for _ in range(n - 1):
        for i in range(n - 1):
            T, Z = _swap_step(T, Z, i)
    return T, Z


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def _klein_core(A, B, n_x: int):
    """Return (F, P, eigenvalues) from Klein.

    The primal pass uses complex Schur (no autodiff rule in jax 0.9.x). The
    JVP rule uses implicit differentiation on the Klein equations for (F, P)
    and passes a zero tangent for eigenvalues, which is a diagnostic only.
    ``n_stable`` is computed by the caller from ``eigenvalues`` so its
    discrete-valued tangent does not appear in the custom-JVP signature.
    """
    A = jnp.asarray(A)
    B = jnp.asarray(B)
    n = A.shape[0]
    n_y = n - n_x

    M = jnp.linalg.solve(A, B)
    T, Z = jax.scipy.linalg.schur(M, output="complex")
    T, Z = _reorder_schur_ascending(T, Z)

    eigenvalues = jnp.diag(T)

    Z11 = Z[:n_x, :n_x]
    Z21 = Z[n_x:, :n_x]
    T11 = T[:n_x, :n_x]

    if n_x == 0:
        F = jnp.zeros((0, 0))
        P = jnp.zeros((n_y, 0))
    else:
        Z11_inv = jnp.linalg.inv(Z11)
        F = jnp.real(Z11 @ T11 @ Z11_inv)
        if n_y == 0:
            P = jnp.zeros((0, n_x))
        else:
            P = jnp.real(Z21 @ Z11_inv)
    return F, P, eigenvalues


@_klein_core.defjvp
def _klein_core_jvp(n_x: int, primals, tangents):
    """JVP via implicit differentiation of the Klein equations.

    At the solution, F and P satisfy
        M_x F = N_x,  M_y F = N_y
    where M_x = A_xx + A_xy P, M_y = A_yx + A_yy P,
    N_x = B_xx + B_xy P,       N_y = B_yx + B_yy P.

    Differentiating both equations and grouping (dF, dP) on the left:
        M_x dF + A_xy dP F - B_xy dP = -dA_xx F - dA_xy P F + dB_xx + dB_xy P
        M_y dF + A_yy dP F - B_yy dP = -dA_yx F - dA_yy P F + dB_yx + dB_yy P
    Vectorize column-major with vec(X Y Z) = (Z^T kron X) vec(Y); solve.
    """
    A, B = primals
    dA, dB = tangents
    F, P, eigenvalues = _klein_core(A, B, n_x)
    n = A.shape[0]
    n_y = n - n_x
    zero_eig = jnp.zeros_like(eigenvalues)

    if n_x == 0:
        return (
            (F, P, eigenvalues),
            (jnp.zeros_like(F), jnp.zeros_like(P), zero_eig),
        )

    A_xy = A[:n_x, n_x:]
    A_yy = A[n_x:, n_x:]
    B_xy = B[:n_x, n_x:]
    B_yy = B[n_x:, n_x:]

    M_x = A[:n_x, :n_x] + A_xy @ P
    M_y = A[n_x:, :n_x] + A_yy @ P

    dA_xx = dA[:n_x, :n_x]; dA_xy = dA[:n_x, n_x:]
    dA_yx = dA[n_x:, :n_x]; dA_yy = dA[n_x:, n_x:]
    dB_xx = dB[:n_x, :n_x]; dB_xy = dB[:n_x, n_x:]
    dB_yx = dB[n_x:, :n_x]; dB_yy = dB[n_x:, n_x:]

    RHS_x = -dA_xx @ F - dA_xy @ P @ F + dB_xx + dB_xy @ P
    RHS_y = -dA_yx @ F - dA_yy @ P @ F + dB_yx + dB_yy @ P

    I_nx = jnp.eye(n_x)
    F_T = F.T

    if n_y == 0:
        operator = jnp.kron(I_nx, M_x)
        rhs = RHS_x.T.reshape(-1)
        dF_vec = jnp.linalg.solve(operator, rhs)
        dF = dF_vec.reshape(n_x, n_x).T
        dP = jnp.zeros_like(P)
        return (
            (F, P, eigenvalues),
            (dF, dP, zero_eig),
        )

    top_left = jnp.kron(I_nx, M_x)
    top_right = jnp.kron(F_T, A_xy) - jnp.kron(I_nx, B_xy)
    bot_left = jnp.kron(I_nx, M_y)
    bot_right = jnp.kron(F_T, A_yy) - jnp.kron(I_nx, B_yy)

    operator = jnp.block([
        [top_left, top_right],
        [bot_left, bot_right],
    ])
    rhs = jnp.concatenate([RHS_x.T.reshape(-1), RHS_y.T.reshape(-1)])
    sol = jnp.linalg.solve(operator, rhs)

    dF = sol[:n_x * n_x].reshape(n_x, n_x).T
    dP = sol[n_x * n_x:].reshape(n_x, n_y).T
    return (
        (F, P, eigenvalues),
        (dF, dP, zero_eig),
    )


def solve_klein_jax(A, B, n_predetermined: int) -> KleinSolutionJax:
    """Solve ``A E_t s_{t+1} = B s_t`` by reducing to standard Schur on JAX.

    State ordering: top ``n_predetermined`` rows of ``s`` are predetermined,
    the rest are forward-looking. Gradient flow is supplied by a custom JVP
    on the implicit Klein equations; the Schur path is used only in the
    primal computation.
    """
    n_x = int(n_predetermined)
    A = jnp.asarray(A)
    B = jnp.asarray(B)
    F, P, eigenvalues = _klein_core(A, B, n_x)
    n_stable = jnp.sum(jnp.abs(eigenvalues) < 1.0)
    bk_satisfied = n_stable == n_x

    return KleinSolutionJax(
        F=F,
        P=P,
        eigenvalues=eigenvalues,
        n_stable=n_stable,
        n_predetermined=n_x,
        bk_satisfied=bk_satisfied,
    )
