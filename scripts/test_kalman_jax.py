#!/usr/bin/env python3
"""Smoke test for the JAX Kalman filter log-likelihood (lib/kalman_jax.py).

Two checks on an AR(1) + measurement-noise state-space model:
    x_{t+1} = rho * x_t + sigma_eta * eta_t,   eta ~ N(0, 1)
    y_t     = x_t + sigma_eps * eps_t,         eps ~ N(0, 1)

1. Numerical match: kalman_loglik agrees with a hand-coded NumPy Kalman
   to ~1e-10 on simulated observations.
2. Differentiability: jax.grad of log p(y; rho, sigma_eta, sigma_eps)
   matches a 5-point central finite-difference reference.

Exit code 0 = all PASS, 1 = any FAIL.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.kalman_jax import kalman_loglik

jax.config.update("jax_enable_x64", True)


def simulate_ar1_obs(rho: float, sigma_eta: float, sigma_eps: float,
                     T_n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = 0.0
    obs = np.zeros(T_n)
    for t in range(T_n):
        x = rho * x + sigma_eta * rng.standard_normal()
        obs[t] = x + sigma_eps * rng.standard_normal()
    return obs


def kalman_loglik_numpy(rho: float, sigma_eta: float, sigma_eps: float,
                        y: np.ndarray) -> float:
    """Reference NumPy Kalman log-likelihood matching kalman_loglik's algebra."""
    Q = sigma_eta ** 2
    S = sigma_eps ** 2
    # Stationary initial variance: P0 = rho^2 P0 + Q  ->  P0 = Q / (1 - rho^2).
    m = 0.0
    P = Q / (1.0 - rho ** 2)
    ll = 0.0
    log_2pi = float(np.log(2.0 * np.pi))
    for y_t in y:
        m_pred = rho * m
        P_pred = rho ** 2 * P + Q
        v = y_t - m_pred
        F = P_pred + S
        K = P_pred / F
        m = m_pred + K * v
        P = P_pred - K * P_pred
        ll += -0.5 * (log_2pi + np.log(F) + v * v / F)
    return ll


def state_space_jax(rho, sigma_eta, sigma_eps):
    T_mat = jnp.array([[rho]])
    R_mat = jnp.array([[sigma_eta]])
    Q_mat = jnp.array([[1.0]])
    H_mat = jnp.array([[1.0]])
    S_mat = jnp.array([[sigma_eps ** 2]])
    return T_mat, R_mat, Q_mat, H_mat, S_mat


THETA = dict(rho=0.7, sigma_eta=0.5, sigma_eps=0.3)
T_N = 200
SEED = 12


def test_match_numpy_reference():
    print("Test 1: JAX Kalman log-lik matches NumPy reference")
    y = simulate_ar1_obs(**THETA, T_n=T_N, seed=SEED)
    ll_np = kalman_loglik_numpy(THETA["rho"], THETA["sigma_eta"],
                                THETA["sigma_eps"], y)
    ss = state_space_jax(THETA["rho"], THETA["sigma_eta"], THETA["sigma_eps"])
    ll_jx = float(kalman_loglik(*ss, jnp.asarray(y)))
    diff = abs(ll_jx - ll_np)
    ok = diff < 1e-10
    print(f"  [{'PASS' if ok else 'FAIL'}] ll_jax={ll_jx:.10f}, "
          f"ll_np={ll_np:.10f}, |diff|={diff:.2e}")
    return ok


def test_differentiability():
    print("\nTest 2: jax.grad of Kalman log-lik matches finite-difference")
    y = simulate_ar1_obs(**THETA, T_n=T_N, seed=SEED)
    theta0 = jnp.array([THETA["rho"], THETA["sigma_eta"], THETA["sigma_eps"]])

    def loglik_jax(theta):
        rho, se, sp = theta
        ss = state_space_jax(rho, se, sp)
        return kalman_loglik(*ss, jnp.asarray(y))

    grad_jax = np.asarray(jax.grad(loglik_jax)(theta0))

    def loglik_np(theta):
        return kalman_loglik_numpy(float(theta[0]), float(theta[1]),
                                   float(theta[2]), y)

    theta_np = np.asarray(theta0)
    eps = 1e-5
    grad_fd = np.zeros_like(theta_np)
    for k in range(len(theta_np)):
        thp = theta_np.copy(); thp[k] += eps
        thpp = theta_np.copy(); thpp[k] += 2 * eps
        thm = theta_np.copy(); thm[k] -= eps
        thmm = theta_np.copy(); thmm[k] -= 2 * eps
        grad_fd[k] = (
            -loglik_np(thpp) + 8 * loglik_np(thp)
            - 8 * loglik_np(thm) + loglik_np(thmm)
        ) / (12 * eps)

    max_diff = float(np.max(np.abs(grad_jax - grad_fd)))
    rel = max_diff / (float(np.max(np.abs(grad_fd))) + 1e-12)
    finite = bool(np.all(np.isfinite(grad_jax)))
    ok = finite and rel < 1e-7
    print(f"  [{'PASS' if ok else 'FAIL'}] grad finite={finite}, "
          f"max |grad_jax - grad_fd|={max_diff:.2e}, rel={rel:.2e}")
    for n, gj, gf in zip(("rho", "sigma_eta", "sigma_eps"), grad_jax, grad_fd):
        print(f"    d log p / d {n:10s}: jax={gj:+.6e}  fd={gf:+.6e}")
    return ok


def main():
    t1 = test_match_numpy_reference()
    t2 = test_differentiability()
    overall = t1 and t2
    print("\n" + "=" * 64)
    print("OVERALL:", "PASS" if overall else "FAIL")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
