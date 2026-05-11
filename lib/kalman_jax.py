"""JAX-native Kalman filter log-likelihood for linear Gaussian state-space.

State equation:    x_{t+1} = T_mat x_t + R_mat eta_t,   eta ~ N(0, Q_mat)
Observation eq:    y_t     = H_mat x_t + eps_t,         eps ~ N(0, S_mat)

The function ``kalman_loglik`` returns the scalar log marginal likelihood
``sum_t log p(y_t | y_{1:t-1}, theta)`` and is fully differentiable through
its state-space arguments via standard JAX autodiff (no custom JVP). The
time recursion is implemented with ``jax.lax.scan`` so the trace is
bounded by state-space dimension, not by series length.

Two initial-state options:
- Pass ``m0``, ``P0`` explicitly.
- Leave both as ``None`` and the function solves the discrete Lyapunov
  equation ``P0 = T P0 T^T + R Q R^T`` (requires ``T_mat`` Schur-stable,
  i.e. all eigenvalues strictly inside the unit circle).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def discrete_lyapunov(T_mat: jnp.ndarray, W: jnp.ndarray) -> jnp.ndarray:
    """Solve ``X = T_mat X T_mat^T + W`` by vectorization.

    Closed-form solution: ``vec(X) = (I - T_mat kron T_mat)^{-1} vec(W)``
    in column-major vec convention. ``T_mat`` must have spectral radius
    strictly less than 1; the caller is responsible for that.
    """
    n = T_mat.shape[0]
    op = jnp.eye(n * n) - jnp.kron(T_mat, T_mat)
    rhs = W.T.reshape(-1)
    vec_X = jnp.linalg.solve(op, rhs)
    return vec_X.reshape(n, n).T


def kalman_loglik(
    T_mat: jnp.ndarray,
    R_mat: jnp.ndarray,
    Q_mat: jnp.ndarray,
    H_mat: jnp.ndarray,
    S_mat: jnp.ndarray,
    observations: jnp.ndarray,
    m0: jnp.ndarray | None = None,
    P0: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Scalar log marginal likelihood under the linear Gaussian model.

    Args:
        T_mat: (ds, ds) state transition.
        R_mat: (ds, dq) shock loading; ``dq`` is the number of structural shocks.
        Q_mat: (dq, dq) shock covariance.
        H_mat: (do, ds) observation matrix; ``do`` is the number of observables.
        S_mat: (do, do) observation noise covariance.
        observations: (Tn, do) observed series.
        m0: optional (ds,) initial state mean; defaults to zeros.
        P0: optional (ds, ds) initial state covariance; defaults to the
            stationary covariance from ``discrete_lyapunov(T, R Q R^T)``.
    """
    Y = jnp.asarray(observations)
    if Y.ndim == 1:
        Y = Y[:, None]
    ds = T_mat.shape[0]
    do = H_mat.shape[0]

    if m0 is None:
        m0 = jnp.zeros(ds)
    if P0 is None:
        W = R_mat @ Q_mat @ R_mat.T
        P0 = discrete_lyapunov(T_mat, W)

    log_2pi = jnp.log(2.0 * jnp.pi)
    RQRt = R_mat @ Q_mat @ R_mat.T

    def step(carry, y_t):
        m, P, ll = carry
        m_pred = T_mat @ m
        P_pred = T_mat @ P @ T_mat.T + RQRt
        v = y_t - H_mat @ m_pred
        F = H_mat @ P_pred @ H_mat.T + S_mat
        F = 0.5 * (F + F.T)
        L = jnp.linalg.cholesky(F)
        u = jax.scipy.linalg.cho_solve((L, True), v)
        logdet_F = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        ll_inc = -0.5 * (do * log_2pi + logdet_F + jnp.dot(v, u))
        # Kalman gain via the same Cholesky: K = P_pred H^T F^{-1}.
        HP = H_mat @ P_pred
        K_T = jax.scipy.linalg.cho_solve((L, True), HP)
        K = K_T.T
        m_new = m_pred + K @ v
        P_new = P_pred - K @ HP
        P_new = 0.5 * (P_new + P_new.T)
        return (m_new, P_new, ll + ll_inc), None

    (_, _, ll_total), _ = jax.lax.scan(step, (m0, P0, jnp.asarray(0.0)), Y)
    return ll_total
