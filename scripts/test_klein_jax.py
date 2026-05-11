#!/usr/bin/env python3
"""Smoke test for the JAX Klein QZ port (lib/perturbation_jax.py).

Three checks on the NK pencil from dsge/nkdsge/run.py:

1. Numerical match: solve_klein_jax F and P agree with scipy solve_klein on
   both the monetary and demand shock pencils.
2. Closed-form match: solve_klein_jax recovers (psi_y, psi_pi) from the
   method-of-undetermined-coefficients solution.
3. Differentiability: jax.grad of psi_y w.r.t. (sigma, beta, phi_pi, phi_y,
   kappa) is finite and matches a 5-point central finite-difference
   reference (computed via scipy solve_klein).

Exit code 0 = all PASS; 1 = any FAIL.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.perturbation import solve_klein
from lib.perturbation_jax import solve_klein_jax

jax.config.update("jax_enable_x64", True)


def build_AB(sigma, beta, phi_pi, phi_y, kappa, rho_shock, shock_kind, use_jax):
    """Build the 3x3 NK (A, B) pencil. Mirrors klein_qz_nk in dsge/nkdsge/run.py."""
    xp = jnp if use_jax else np
    if shock_kind == "monetary":
        v_in_is = 1.0 / sigma
    elif shock_kind == "demand":
        v_in_is = -1.0
    else:
        raise ValueError(shock_kind)
    A = xp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0 / sigma],
            [0.0, 0.0, beta],
        ]
    )
    B = xp.array(
        [
            [rho_shock, 0.0, 0.0],
            [v_in_is, 1.0 + phi_y / sigma, phi_pi / sigma],
            [0.0, -kappa, 1.0],
        ]
    )
    return A, B


def analytical_psi(sigma, beta, phi_pi, phi_y, kappa, rho, shock_kind):
    """Closed-form (psi_y, psi_pi) from method of undetermined coefficients.

    Reproduces solve_nk_model / solve_nk_demand_shock from dsge/nkdsge/run.py.
    """
    denom_pc = 1.0 - beta * rho
    coeff = (1.0 - rho) + phi_y / sigma + (phi_pi - rho) * kappa / (sigma * denom_pc)
    if shock_kind == "monetary":
        psi_y = -1.0 / (sigma * coeff)
    else:
        psi_y = 1.0 / coeff
    psi_pi = kappa * psi_y / denom_pc
    return psi_y, psi_pi


THETA = dict(sigma=1.0, beta=0.99, phi_pi=1.5, phi_y=0.125, kappa=0.3)
RHO_V = 0.5
RHO_D = 0.8


def test_match_scipy():
    print("Test 1: JAX Klein matches scipy Klein on NK pencil")
    passes = []
    for shock_kind, rho in [("monetary", RHO_V), ("demand", RHO_D)]:
        args = (THETA["sigma"], THETA["beta"], THETA["phi_pi"],
                THETA["phi_y"], THETA["kappa"], rho, shock_kind)
        A_np, B_np = build_AB(*args, use_jax=False)
        A_jx, B_jx = build_AB(*args, use_jax=True)
        sol_scipy = solve_klein(A_np, B_np, n_predetermined=1)
        sol_jax = solve_klein_jax(A_jx, B_jx, n_predetermined=1)
        F_diff = float(np.max(np.abs(np.asarray(sol_jax.F) - sol_scipy.F)))
        P_diff = float(np.max(np.abs(np.asarray(sol_jax.P) - sol_scipy.P)))
        ok = F_diff < 1e-10 and P_diff < 1e-10
        passes.append(ok)
        verdict = "PASS" if ok else "FAIL"
        print(f"  [{verdict}] {shock_kind:8s}: max |F_diff|={F_diff:.2e}, "
              f"max |P_diff|={P_diff:.2e}")
    return all(passes)


def test_match_closed_form():
    print("\nTest 2: JAX Klein matches closed-form (psi_y, psi_pi)")
    passes = []
    for shock_kind, rho in [("monetary", RHO_V), ("demand", RHO_D)]:
        psi_y_an, psi_pi_an = analytical_psi(
            THETA["sigma"], THETA["beta"], THETA["phi_pi"], THETA["phi_y"],
            THETA["kappa"], rho, shock_kind
        )
        args = (THETA["sigma"], THETA["beta"], THETA["phi_pi"],
                THETA["phi_y"], THETA["kappa"], rho, shock_kind)
        A_jx, B_jx = build_AB(*args, use_jax=True)
        sol = solve_klein_jax(A_jx, B_jx, n_predetermined=1)
        psi_y_jx = float(sol.P[0, 0])
        psi_pi_jx = float(sol.P[1, 0])
        diff = max(abs(psi_y_jx - psi_y_an), abs(psi_pi_jx - psi_pi_an))
        ok = diff < 1e-12
        passes.append(ok)
        verdict = "PASS" if ok else "FAIL"
        print(f"  [{verdict}] {shock_kind:8s}: psi_y JAX={psi_y_jx:+.8f} "
              f"vs analytic={psi_y_an:+.8f} (max diff {diff:.2e})")
    return all(passes)


def test_differentiability():
    print("\nTest 3: jax.grad through solve_klein_jax matches finite-difference")
    rho = RHO_V
    shock_kind = "monetary"
    theta0 = jnp.array([THETA["sigma"], THETA["beta"], THETA["phi_pi"],
                        THETA["phi_y"], THETA["kappa"]])

    def psi_y_jax(theta):
        sigma, beta, phi_pi, phi_y, kappa = theta
        A, B = build_AB(sigma, beta, phi_pi, phi_y, kappa, rho, shock_kind,
                        use_jax=True)
        sol = solve_klein_jax(A, B, n_predetermined=1)
        return sol.P[0, 0]

    grad_jax = np.asarray(jax.grad(psi_y_jax)(theta0))

    def psi_y_np(theta_np):
        sigma, beta, phi_pi, phi_y, kappa = theta_np
        A, B = build_AB(sigma, beta, phi_pi, phi_y, kappa, rho, shock_kind,
                        use_jax=False)
        sol = solve_klein(A, B, n_predetermined=1)
        return sol.P[0, 0]

    theta_np = np.asarray(theta0)
    eps = 1e-5
    grad_fd = np.zeros_like(theta_np)
    for k in range(len(theta_np)):
        thp = theta_np.copy(); thp[k] += eps
        thpp = theta_np.copy(); thpp[k] += 2 * eps
        thm = theta_np.copy(); thm[k] -= eps
        thmm = theta_np.copy(); thmm[k] -= 2 * eps
        grad_fd[k] = (
            -psi_y_np(thpp) + 8 * psi_y_np(thp)
            - 8 * psi_y_np(thm) + psi_y_np(thmm)
        ) / (12 * eps)

    max_diff = float(np.max(np.abs(grad_jax - grad_fd)))
    rel_diff = max_diff / (float(np.max(np.abs(grad_fd))) + 1e-12)
    finite = bool(np.all(np.isfinite(grad_jax)))
    ok = finite and rel_diff < 1e-6
    verdict = "PASS" if ok else "FAIL"
    print(f"  [{verdict}] grad finite={finite}, max |grad_jax - grad_fd|={max_diff:.2e}, "
          f"rel={rel_diff:.2e}")
    names = ["sigma", "beta", "phi_pi", "phi_y", "kappa"]
    for name, gj, gf in zip(names, grad_jax, grad_fd):
        print(f"    d psi_y / d {name:7s}: jax={gj:+.6e}  fd={gf:+.6e}")
    return ok


def main():
    t1 = test_match_scipy()
    t2 = test_match_closed_form()
    t3 = test_differentiability()
    overall = t1 and t2 and t3
    print("\n" + "=" * 64)
    print("OVERALL:", "PASS" if overall else "FAIL")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
