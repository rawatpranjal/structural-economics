#!/usr/bin/env python3
"""End-to-end smoke test: NK model -> Klein -> state-space -> Kalman log-lik.

The 2-shock NK pencil is 4x4 with state ordering (v, d, y, pi) where v is
the monetary shock, d the demand shock (both predetermined), and (y, pi)
forward-looking. Klein gives F (2x2 transition for (v, d)) and P (2x2
policy mapping shocks to (y, pi)). Observables are (y, pi, i) where the
nominal rate comes from the Taylor rule.

Two checks:
1. Numerical: JAX log-likelihood agrees with a scipy-Klein + NumPy-Kalman
   reference on the same simulated series.
2. Differentiability: jax.grad(loglik)(theta) is finite and matches
   5-point central FD across all 9 structural parameters.

This is the most consequential check before building the Bayesian DSGE
tutorial: it confirms Klein's custom JVP and Kalman's standard autodiff
compose cleanly end to end.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.kalman_jax import kalman_loglik
from lib.perturbation import solve_klein
from lib.perturbation_jax import solve_klein_jax

jax.config.update("jax_enable_x64", True)


PARAM_NAMES = ["sigma", "beta", "phi_pi", "phi_y", "kappa",
               "sigma_v", "rho_v", "sigma_d", "rho_d"]
THETA0 = np.array([1.0, 0.99, 1.5, 0.125, 0.3, 0.01, 0.5, 0.01, 0.8])
MEAS_STD = 1e-3
T_N = 120
SEED = 31


def build_AB(sigma, beta, phi_pi, phi_y, kappa, rho_v, rho_d, use_jax=False):
    """Two-shock NK pencil: state s = (v, d, y, pi), n_predetermined=2."""
    xp = jnp if use_jax else np
    A = xp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0 / sigma],
        [0.0, 0.0, 0.0, beta],
    ])
    B = xp.array([
        [rho_v, 0.0, 0.0, 0.0],
        [0.0, rho_d, 0.0, 0.0],
        [1.0 / sigma, -1.0, 1.0 + phi_y / sigma, phi_pi / sigma],
        [0.0, 0.0, -kappa, 1.0],
    ])
    return A, B


def state_space_jax(theta):
    """Return (T_mat, R_mat, Q_mat, H_mat, S_mat) from JAX Klein."""
    sigma, beta, phi_pi, phi_y, kappa, s_v, rho_v, s_d, rho_d = theta
    A, B = build_AB(sigma, beta, phi_pi, phi_y, kappa, rho_v, rho_d, use_jax=True)
    sol = solve_klein_jax(A, B, n_predetermined=2)
    F = sol.F  # (2, 2) transition of (v, d)
    P = sol.P  # (2, 2) policy mapping (v, d) -> (y, pi)
    T_mat = F
    R_mat = jnp.array([[s_v, 0.0], [0.0, s_d]])
    Q_mat = jnp.eye(2)
    # H rows: y, pi, i = phi_pi * pi + phi_y * y + v
    H_mat = jnp.stack([
        P[0],                                        # y_t  = P[0,0] v + P[0,1] d
        P[1],                                        # pi_t = P[1,0] v + P[1,1] d
        jnp.array([phi_pi * P[1, 0] + phi_y * P[0, 0] + 1.0,
                   phi_pi * P[1, 1] + phi_y * P[0, 1]]),
    ])
    S_mat = (MEAS_STD ** 2) * jnp.eye(3)
    return T_mat, R_mat, Q_mat, H_mat, S_mat


def state_space_numpy(theta):
    """Same construction in NumPy / scipy for the reference path."""
    sigma, beta, phi_pi, phi_y, kappa, s_v, rho_v, s_d, rho_d = theta
    A, B = build_AB(sigma, beta, phi_pi, phi_y, kappa, rho_v, rho_d, use_jax=False)
    sol = solve_klein(A, B, n_predetermined=2)
    F = sol.F
    P = sol.P
    T_mat = F
    R_mat = np.diag([s_v, s_d])
    Q_mat = np.eye(2)
    H_mat = np.vstack([
        P[0],
        P[1],
        np.array([phi_pi * P[1, 0] + phi_y * P[0, 0] + 1.0,
                  phi_pi * P[1, 1] + phi_y * P[0, 1]]),
    ])
    S_mat = (MEAS_STD ** 2) * np.eye(3)
    return T_mat, R_mat, Q_mat, H_mat, S_mat


def discrete_lyapunov_np(T_mat, W):
    n = T_mat.shape[0]
    op = np.eye(n * n) - np.kron(T_mat, T_mat)
    return np.linalg.solve(op, W.T.reshape(-1)).reshape(n, n).T


def kalman_loglik_np(T_mat, R_mat, Q_mat, H_mat, S_mat, Y):
    ds = T_mat.shape[0]
    do = H_mat.shape[0]
    W = R_mat @ Q_mat @ R_mat.T
    m = np.zeros(ds)
    P = discrete_lyapunov_np(T_mat, W)
    RQRt = W
    ll = 0.0
    log_2pi = float(np.log(2.0 * np.pi))
    for y_t in Y:
        m_pred = T_mat @ m
        P_pred = T_mat @ P @ T_mat.T + RQRt
        v = y_t - H_mat @ m_pred
        F = H_mat @ P_pred @ H_mat.T + S_mat
        F = 0.5 * (F + F.T)
        L = np.linalg.cholesky(F)
        u = np.linalg.solve(L.T, np.linalg.solve(L, v))
        logdet_F = 2.0 * np.sum(np.log(np.diag(L)))
        ll += -0.5 * (do * log_2pi + logdet_F + v @ u)
        HP = H_mat @ P_pred
        K_T = np.linalg.solve(L.T, np.linalg.solve(L, HP))
        K = K_T.T
        m = m_pred + K @ v
        P = P_pred - K @ HP
        P = 0.5 * (P + P.T)
    return float(ll)


def simulate_observations(theta, T_n: int, seed: int):
    rng = np.random.default_rng(seed)
    T_mat, R_mat, Q_mat, H_mat, S_mat = state_space_numpy(theta)
    ds = T_mat.shape[0]
    do = H_mat.shape[0]
    x = np.zeros(ds)
    Y = np.zeros((T_n, do))
    chol_Q = np.linalg.cholesky(R_mat @ Q_mat @ R_mat.T + 1e-18 * np.eye(ds))
    chol_S = np.linalg.cholesky(S_mat)
    for t in range(T_n):
        x = T_mat @ x + chol_Q @ rng.standard_normal(ds)
        Y[t] = H_mat @ x + chol_S @ rng.standard_normal(do)
    return Y


def test_match_numpy():
    print("Test 1: JAX log-lik matches scipy+NumPy reference on 2-shock NK")
    Y = simulate_observations(THETA0, T_N, SEED)
    ll_jx = float(kalman_loglik(*state_space_jax(jnp.asarray(THETA0)), jnp.asarray(Y)))
    ll_np = kalman_loglik_np(*state_space_numpy(THETA0), Y)
    diff = abs(ll_jx - ll_np)
    ok = diff < 1e-8
    print(f"  [{'PASS' if ok else 'FAIL'}] ll_jax={ll_jx:.6f}, ll_np={ll_np:.6f}, |diff|={diff:.2e}")
    return ok


def test_differentiability():
    print("\nTest 2: jax.grad of NK log-lik finite and matches FD")
    Y = simulate_observations(THETA0, T_N, SEED)
    theta0_jx = jnp.asarray(THETA0)

    def loglik_jax(theta):
        return kalman_loglik(*state_space_jax(theta), jnp.asarray(Y))

    grad_jax = np.asarray(jax.grad(loglik_jax)(theta0_jx))

    def loglik_np(theta):
        return kalman_loglik_np(*state_space_numpy(theta), Y)

    eps = 1e-5
    grad_fd = np.zeros_like(THETA0)
    for k in range(len(THETA0)):
        thp = THETA0.copy(); thp[k] += eps
        thpp = THETA0.copy(); thpp[k] += 2 * eps
        thm = THETA0.copy(); thm[k] -= eps
        thmm = THETA0.copy(); thmm[k] -= 2 * eps
        grad_fd[k] = (
            -loglik_np(thpp) + 8 * loglik_np(thp)
            - 8 * loglik_np(thm) + loglik_np(thmm)
        ) / (12 * eps)

    max_diff = float(np.max(np.abs(grad_jax - grad_fd)))
    rel = max_diff / (float(np.max(np.abs(grad_fd))) + 1e-12)
    finite = bool(np.all(np.isfinite(grad_jax)))
    ok = finite and rel < 1e-4
    print(f"  [{'PASS' if ok else 'FAIL'}] grad finite={finite}, "
          f"max |grad_jax - grad_fd|={max_diff:.2e}, rel={rel:.2e}")
    for n, gj, gf in zip(PARAM_NAMES, grad_jax, grad_fd):
        print(f"    d log p / d {n:8s}: jax={gj:+.6e}  fd={gf:+.6e}")
    return ok


def main():
    t1 = test_match_numpy()
    t2 = test_differentiability()
    overall = t1 and t2
    print("\n" + "=" * 70)
    print("OVERALL:", "PASS" if overall else "FAIL")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
