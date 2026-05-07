#!/usr/bin/env python3
"""Differentiated-products merger pricing under calibrated demand systems.

Calibrates logit, linear, and log-linear demand to one pre-merger market,
changes product ownership, and solves Bertrand-Nash counterfactual prices.
The same run also compares UPP, GUPPI, CMCR, and efficiency thresholds.

Reference: Werden and Froeb (1994), Farrell and Shapiro (2010).
"""
import sys
from pathlib import Path

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


# =========================================================================
# Ownership matrix
# =========================================================================

def ownership_matrix(p2f: np.ndarray) -> np.ndarray:
    """Ownership matrix: Omega[j,k] = 1 if products j and k belong to same firm."""
    J = len(p2f)
    return (p2f[:, None] == p2f[None, :]).astype(float)


# =========================================================================
# Demand System 1: Logit
# =========================================================================

def shares_logit(p: np.ndarray, alpha: float, xi: np.ndarray) -> np.ndarray:
    """Logit shares: s_j = exp(xi_j + alpha*p_j) / (1 + sum exp(...))."""
    v = np.exp(xi + alpha * p)
    denom = 1.0 + np.sum(v)
    return v / denom


def jacobian_logit(p: np.ndarray, alpha: float, xi: np.ndarray) -> np.ndarray:
    """Jacobian ds/dp for logit demand."""
    s = shares_logit(p, alpha, xi)
    J = -alpha * np.outer(s, s)
    np.fill_diagonal(J, alpha * s * (1.0 - s))
    return J


def foc_logit(p: np.ndarray, mc: np.ndarray, alpha: float,
              xi: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Bertrand-Nash FOC: s + (Omega * dsdp') (p - mc) = 0."""
    s = shares_logit(p, alpha, xi)
    dsdp = jacobian_logit(p, alpha, xi)
    return s + (omega * dsdp.T) @ (p - mc)


def cs_logit(p: np.ndarray, alpha: float, xi: np.ndarray, M: float) -> float:
    """Consumer surplus (logit): CS = -M/alpha * ln(1 + sum exp(xi + alpha*p))."""
    v = np.exp(xi + alpha * p)
    return -M / alpha * np.log(1.0 + np.sum(v))


# =========================================================================
# Demand System 2: Linear Demand
# =========================================================================

def shares_linear(p: np.ndarray, a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Linear demand: q = a - B @ p, returned as shares (q / M)."""
    q = a - B @ p
    return np.maximum(q, 1e-12)


def jacobian_linear(p: np.ndarray, a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Jacobian dq/dp = -B for linear demand."""
    return -B


def foc_linear(p: np.ndarray, mc: np.ndarray, a: np.ndarray,
               B: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Bertrand-Nash FOC for linear demand."""
    q = shares_linear(p, a, B)
    dqdp = jacobian_linear(p, a, B)
    return q + (omega * dqdp.T) @ (p - mc)


def cs_linear(p: np.ndarray, a: np.ndarray, B: np.ndarray,
              p_choke: np.ndarray) -> float:
    """Consumer surplus for linear demand (area under demand curve above price).

    CS = 0.5 * (p_choke - p)' B_diag (p_choke - p) approximately, using the
    quadratic form: CS = 0.5 * q' B^{-1} q.
    """
    q = shares_linear(p, a, B)
    try:
        B_inv = np.linalg.inv(B)
        return 0.5 * q @ B_inv @ q
    except np.linalg.LinAlgError:
        return 0.5 * np.sum(q ** 2 / np.diag(B))


# =========================================================================
# Demand System 3: Log-Linear Demand
# =========================================================================

def shares_loglinear(p: np.ndarray, a_ll: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Log-linear demand: ln q_j = a_j + sum_k E_{jk} ln p_k.

    So q_j = exp(a_j) * prod_k p_k^{E_{jk}}.
    """
    log_q = a_ll + E @ np.log(p)
    return np.exp(log_q)


def jacobian_loglinear(p: np.ndarray, a_ll: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Jacobian dq/dp for log-linear demand: dq_j/dp_k = q_j * E_{jk} / p_k."""
    q = shares_loglinear(p, a_ll, E)
    return (q[:, None] * E) / p[None, :]


def foc_loglinear(p: np.ndarray, mc: np.ndarray, a_ll: np.ndarray,
                  E: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Bertrand-Nash FOC for log-linear demand."""
    q = shares_loglinear(p, a_ll, E)
    dqdp = jacobian_loglinear(p, a_ll, E)
    return q + (omega * dqdp.T) @ (p - mc)


def foc_loglinear_logp(log_p: np.ndarray, mc: np.ndarray, a_ll: np.ndarray,
                       E: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """FOC in log-price space (ensures prices stay positive during solver iterations)."""
    p = np.exp(log_p)
    return foc_loglinear(p, mc, a_ll, E, omega)


def cs_loglinear(p: np.ndarray, p_high: np.ndarray, a_ll: np.ndarray,
                 E: np.ndarray, n_steps: int = 200) -> float:
    """Consumer surplus for log-linear demand via numerical integration.

    Integrate q(t) dp from p to p_high along a linear price path.
    """
    cs = 0.0
    for i in range(n_steps):
        t0 = i / n_steps
        t1 = (i + 1) / n_steps
        t_mid = 0.5 * (t0 + t1)
        p_mid = p + t_mid * (p_high - p)
        q_mid = shares_loglinear(p_mid, a_ll, E)
        dp = (p_high - p) / n_steps
        cs += np.sum(q_mid * dp)
    return cs


# =========================================================================
# Calibration
# =========================================================================

def calibrate_logit(shares_obs: np.ndarray, prices_obs: np.ndarray,
                    margins_obs: np.ndarray, p2f: np.ndarray) -> dict:
    """Calibrate logit demand from observed shares, prices, and margins.

    Strategy: use the average margin to pin down alpha, then invert the full
    multi-product FOC system to recover marginal costs exactly.
    """
    omega = ownership_matrix(p2f)
    J = len(shares_obs)
    s0_total = np.sum(shares_obs)

    # Use the average margin across all products to pin down alpha.
    # For a single-product firm j: margin_j = -1 / (alpha * (1 - s_j))
    # => alpha = -1 / (margin_j * (1 - s_j))
    # Average across products rather than relying on one margin:
    alpha_estimates = -1.0 / (margins_obs * (1.0 - shares_obs))
    alpha = np.mean(alpha_estimates)

    # Mean valuations xi (from share inversion)
    xi = np.log(shares_obs / (1.0 - s0_total)) - alpha * prices_obs

    # Recover marginal costs by inverting the full FOC system:
    # s + (Omega * dsdp') (p - mc) = 0  =>  mc = p + inv(Omega * dsdp') s
    dsdp = jacobian_logit(prices_obs, alpha, xi)
    mc = prices_obs + np.linalg.solve(omega * dsdp.T, shares_obs)

    return {"alpha": alpha, "xi": xi, "mc": mc}


def calibrate_linear(shares_obs: np.ndarray, prices_obs: np.ndarray,
                     margins_obs: np.ndarray, p2f: np.ndarray,
                     cross_ratio: float = 0.1) -> dict:
    """Calibrate linear demand: q = a - B @ p.

    Own-price elasticity matched to margins via FOC. Cross-price elasticities
    set as a fraction of own-price effects.
    """
    J = len(shares_obs)
    mc = prices_obs * (1.0 - margins_obs)
    omega = ownership_matrix(p2f)

    # Market size: total quantity normalized
    M = 1.0
    q_obs = shares_obs * M

    # From FOC for single-product firm j: q_j - b_jj * (p_j - mc_j) = 0
    # => b_jj = q_j / (p_j - mc_j)
    # For multi-product firms, need to account for portfolio.
    # Start with single-product FOC for own-price slope:
    markups = prices_obs - mc
    b_own = q_obs / markups

    # Build B matrix: B[j,k] = b_jj on diagonal, cross terms = -cross_ratio * sqrt(b_jj * b_kk)
    B = np.zeros((J, J))
    for j in range(J):
        B[j, j] = b_own[j]
        for k in range(J):
            if k != j:
                B[j, k] = -cross_ratio * np.sqrt(b_own[j] * b_own[k])

    # Adjust own-price slopes for multi-product firms to satisfy FOC exactly
    for j in range(J):
        cross_contrib = 0.0
        for k in range(J):
            if k != j and omega[j, k] == 1:
                cross_contrib += (-B[k, j]) * markups[k]  # dq_k/dp_j * (p_k - mc_k)
        # FOC: q_j + (-b_jj)*(p_j - mc_j) + cross_contrib = 0
        # => b_jj = (q_j + cross_contrib) / (p_j - mc_j)
        B[j, j] = (q_obs[j] + cross_contrib) / markups[j]

    # Intercepts: a = q + B @ p
    a = q_obs + B @ prices_obs

    return {"a": a, "B": B, "mc": mc}


def calibrate_loglinear(shares_obs: np.ndarray, prices_obs: np.ndarray,
                        margins_obs: np.ndarray, p2f: np.ndarray,
                        cross_elas: float = 0.3) -> dict:
    """Calibrate log-linear demand: ln q_j = a_j + sum E_{jk} ln p_k.

    Own-price elasticities from FOC, cross-elasticities set symmetrically.
    """
    J = len(shares_obs)
    mc = prices_obs * (1.0 - margins_obs)
    omega = ownership_matrix(p2f)
    M = 1.0
    q_obs = shares_obs * M
    markups = prices_obs - mc

    # From FOC for firm owning product j only:
    # q_j + (q_j * e_jj / p_j) * (p_j - mc_j) = 0
    # => e_jj = -p_j / (p_j - mc_j)
    e_own = -prices_obs / markups

    # Build elasticity matrix E
    E = np.zeros((J, J))
    for j in range(J):
        E[j, j] = e_own[j]
        for k in range(J):
            if k != j:
                E[j, k] = cross_elas

    # Adjust own-price elasticity for multi-product firms
    for j in range(J):
        cross_contrib = 0.0
        for k in range(J):
            if k != j and omega[j, k] == 1:
                # dq_k/dp_j * (p_k - mc_k) = q_k * E[k,j] / p_j * (p_k - mc_k)
                cross_contrib += q_obs[k] * E[k, j] / prices_obs[j] * markups[k]
        # FOC: q_j + q_j * E[j,j] / p_j * (p_j - mc_j) + cross_contrib = 0
        # => E[j,j] = -(q_j + cross_contrib) * p_j / (q_j * (p_j - mc_j))
        E[j, j] = -(q_obs[j] + cross_contrib) * prices_obs[j] / (q_obs[j] * markups[j])

    # Intercepts: ln q = a + E @ ln p => a = ln q - E @ ln p
    a_ll = np.log(q_obs) - E @ np.log(prices_obs)

    return {"a_ll": a_ll, "E": E, "mc": mc}


# =========================================================================
# Screening Metrics: UPP, GUPPI, CMCR
# =========================================================================

def diversion_ratios_from_jacobian(dqdp: np.ndarray) -> np.ndarray:
    """Diversion ratio D_{j->k} = -(dq_k/dp_j) / (dq_j/dp_j)."""
    J = dqdp.shape[0]
    D = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j != k:
                D[j, k] = -dqdp[k, j] / dqdp[j, j]
    return D


def compute_upp(D: np.ndarray, prices: np.ndarray, mc: np.ndarray,
                p2f_pre: np.ndarray, p2f_post: np.ndarray) -> np.ndarray:
    """Upward Pricing Pressure for each product from the merger.

    UPP_j = sum_{k: newly co-owned} D_{j->k} * (p_k - mc_k)
    """
    J = len(prices)
    upp = np.zeros(J)
    for j in range(J):
        for k in range(J):
            if j != k and p2f_post[j] == p2f_post[k] and p2f_pre[j] != p2f_pre[k]:
                upp[j] += D[j, k] * (prices[k] - mc[k])
    return upp


def compute_guppi(D: np.ndarray, prices: np.ndarray, mc: np.ndarray,
                  p2f_pre: np.ndarray, p2f_post: np.ndarray) -> np.ndarray:
    """GUPPI = UPP_j / p_j (as fraction of price)."""
    upp = compute_upp(D, prices, mc, p2f_pre, p2f_post)
    return upp / prices


def compute_cmcr(D: np.ndarray, prices: np.ndarray, mc: np.ndarray,
                 p2f_pre: np.ndarray, p2f_post: np.ndarray) -> np.ndarray:
    """Compensating Marginal Cost Reduction: the mc reduction that offsets UPP.

    CMCR_j = UPP_j / mc_j (as fraction of marginal cost).
    """
    upp = compute_upp(D, prices, mc, p2f_pre, p2f_post)
    return upp / mc


# =========================================================================
# Welfare Computation
# =========================================================================

def producer_surplus(p: np.ndarray, q: np.ndarray, mc: np.ndarray) -> float:
    """Total producer surplus: sum (p_j - mc_j) * q_j."""
    return np.sum((p - mc) * q)


def fmt_vector(values: np.ndarray, digits: int = 2) -> str:
    """Format a short numeric vector for generated markdown tables."""
    return "[" + ", ".join(f"{float(v):.{digits}f}" for v in values) + "]"


def first_zero_crossing(x: np.ndarray, y: list[float]) -> float:
    """Linearly interpolate the first x value where y crosses zero."""
    y_arr = np.asarray(y, dtype=float)
    for i in range(1, len(x)):
        if (y_arr[i - 1] >= 0 and y_arr[i] <= 0) or (y_arr[i - 1] <= 0 and y_arr[i] >= 0):
            if np.isclose(y_arr[i], y_arr[i - 1]):
                return float(x[i])
            weight = -y_arr[i - 1] / (y_arr[i] - y_arr[i - 1])
            return float(x[i - 1] + weight * (x[i] - x[i - 1]))
    return float("nan")


# =========================================================================
# Main
# =========================================================================

def main():
    # =====================================================================
    # Market Setup: J=6 products, 3 firms (2 products each)
    # =====================================================================
    J = 6
    shares_obs = np.array([0.12, 0.10, 0.15, 0.13, 0.08, 0.07])
    prices_obs = np.array([1.0, 1.2, 0.9, 1.1, 1.3, 1.4])
    margins_obs = np.array([0.40, 0.35, 0.45, 0.40, 0.30, 0.28])
    p2f_pre = np.array([1, 1, 2, 2, 3, 3])  # Firm 1: {0,1}, Firm 2: {2,3}, Firm 3: {4,5}
    M = 1.0  # Market size normalization

    product_names = [f"P{j+1}" for j in range(J)]

    # Post-merger: Firm 1 acquires Firm 2
    p2f_post = np.array([1, 1, 1, 1, 3, 3])

    print("=" * 70)
    print("MERGER PRICING: Demand Systems and Functional Form")
    print("=" * 70)
    print(f"Products: {J}, Firms pre-merger: 3, Firms post-merger: 2")
    print(f"Merger: Firm 1 acquires Firm 2")
    print(f"Observed shares: {shares_obs}")
    print(f"Observed prices: {prices_obs}")
    print(f"Observed margins: {margins_obs}")
    print()

    omega_pre = ownership_matrix(p2f_pre)
    omega_post = ownership_matrix(p2f_post)

    # =====================================================================
    # Calibrate all three demand systems
    # =====================================================================
    cal_logit = calibrate_logit(shares_obs, prices_obs, margins_obs, p2f_pre)
    cal_linear = calibrate_linear(shares_obs, prices_obs, margins_obs, p2f_pre, cross_ratio=0.1)
    cal_loglinear = calibrate_loglinear(shares_obs, prices_obs, margins_obs, p2f_pre, cross_elas=0.15)

    print("--- Logit calibration ---")
    print(f"  alpha = {cal_logit['alpha']:.4f}")
    print(f"  mc = {cal_logit['mc']}")
    print()

    print("--- Linear calibration ---")
    print(f"  Own-price slopes (diag B) = {np.diag(cal_linear['B'])}")
    print(f"  mc = {cal_linear['mc']}")
    print()

    print("--- Log-linear calibration ---")
    print(f"  Own-price elasticities = {np.diag(cal_loglinear['E'])}")
    print(f"  mc = {cal_loglinear['mc']}")
    print()

    # =====================================================================
    # Verify calibration: FOC at observed prices should be ~0
    # =====================================================================
    q_obs = shares_obs * M
    foc_check_logit = foc_logit(prices_obs, cal_logit['mc'], cal_logit['alpha'],
                                cal_logit['xi'], omega_pre)
    foc_check_linear = foc_linear(prices_obs, cal_linear['mc'], cal_linear['a'],
                                  cal_linear['B'], omega_pre)
    foc_check_loglinear = foc_loglinear(prices_obs, cal_loglinear['mc'], cal_loglinear['a_ll'],
                                        cal_loglinear['E'], omega_pre)

    print("FOC verification (max absolute residual):")
    print(f"  Logit:      {np.max(np.abs(foc_check_logit)):.2e}")
    print(f"  Linear:     {np.max(np.abs(foc_check_linear)):.2e}")
    print(f"  Log-linear: {np.max(np.abs(foc_check_loglinear)):.2e}")
    print()

    # =====================================================================
    # Compute pre-merger Jacobians and diversion ratios
    # =====================================================================
    jac_logit = jacobian_logit(prices_obs, cal_logit['alpha'], cal_logit['xi'])
    jac_linear = jacobian_linear(prices_obs, cal_linear['a'], cal_linear['B'])
    jac_loglinear = jacobian_loglinear(prices_obs, cal_loglinear['a_ll'], cal_loglinear['E'])

    div_logit = diversion_ratios_from_jacobian(jac_logit)
    div_linear = diversion_ratios_from_jacobian(jac_linear)
    div_loglinear = diversion_ratios_from_jacobian(jac_loglinear)

    # =====================================================================
    # Screening metrics: UPP, GUPPI, CMCR (each model uses its own mc)
    # =====================================================================
    mc_logit = cal_logit['mc']
    mc_linear = cal_linear['mc']
    mc_loglinear = cal_loglinear['mc']

    upp_logit = compute_upp(div_logit, prices_obs, mc_logit, p2f_pre, p2f_post)
    upp_linear = compute_upp(div_linear, prices_obs, mc_linear, p2f_pre, p2f_post)
    upp_loglinear = compute_upp(div_loglinear, prices_obs, mc_loglinear, p2f_pre, p2f_post)

    guppi_logit = compute_guppi(div_logit, prices_obs, mc_logit, p2f_pre, p2f_post)
    guppi_linear = compute_guppi(div_linear, prices_obs, mc_linear, p2f_pre, p2f_post)
    guppi_loglinear = compute_guppi(div_loglinear, prices_obs, mc_loglinear, p2f_pre, p2f_post)

    cmcr_logit = compute_cmcr(div_logit, prices_obs, mc_logit, p2f_pre, p2f_post)
    cmcr_linear = compute_cmcr(div_linear, prices_obs, mc_linear, p2f_pre, p2f_post)
    cmcr_loglinear = compute_cmcr(div_loglinear, prices_obs, mc_loglinear, p2f_pre, p2f_post)

    print("--- Screening Metrics (merging products 0-3) ---")
    for j in range(4):
        print(f"  Product {j}: UPP = ({upp_logit[j]:.4f}, {upp_linear[j]:.4f}, {upp_loglinear[j]:.4f})  "
              f"GUPPI = ({guppi_logit[j]:.4f}, {guppi_linear[j]:.4f}, {guppi_loglinear[j]:.4f})")
    print()

    # =====================================================================
    # Solve post-merger equilibria for each demand system
    # =====================================================================
    print("Solving post-merger equilibria...")

    # Logit
    p_post_logit = scipy.optimize.fsolve(
        foc_logit, x0=prices_obs * 1.05,
        args=(cal_logit['mc'], cal_logit['alpha'], cal_logit['xi'], omega_post),
        full_output=False
    )
    s_post_logit = shares_logit(p_post_logit, cal_logit['alpha'], cal_logit['xi'])

    # Linear
    p_post_linear = scipy.optimize.fsolve(
        foc_linear, x0=prices_obs * 1.05,
        args=(cal_linear['mc'], cal_linear['a'], cal_linear['B'], omega_post),
        full_output=False
    )
    q_post_linear = shares_linear(p_post_linear, cal_linear['a'], cal_linear['B'])

    # Log-linear (solve in log-price space to keep prices positive)
    logp_post_loglinear = scipy.optimize.fsolve(
        foc_loglinear_logp, x0=np.log(prices_obs * 1.05),
        args=(cal_loglinear['mc'], cal_loglinear['a_ll'], cal_loglinear['E'], omega_post),
        full_output=False
    )
    p_post_loglinear = np.exp(logp_post_loglinear)
    q_post_loglinear = shares_loglinear(p_post_loglinear, cal_loglinear['a_ll'], cal_loglinear['E'])

    print(f"  Logit post-merger prices:      {p_post_logit}")
    print(f"  Linear post-merger prices:     {p_post_linear}")
    print(f"  Log-linear post-merger prices: {p_post_loglinear}")
    print()

    foc_post_logit = foc_logit(
        p_post_logit, cal_logit['mc'], cal_logit['alpha'], cal_logit['xi'], omega_post
    )
    foc_post_linear = foc_linear(
        p_post_linear, cal_linear['mc'], cal_linear['a'], cal_linear['B'], omega_post
    )
    foc_post_loglinear = foc_loglinear(
        p_post_loglinear, cal_loglinear['mc'], cal_loglinear['a_ll'],
        cal_loglinear['E'], omega_post
    )

    # Price changes
    dp_logit = (p_post_logit - prices_obs) / prices_obs * 100
    dp_linear = (p_post_linear - prices_obs) / prices_obs * 100
    dp_loglinear = (p_post_loglinear - prices_obs) / prices_obs * 100

    print("Price changes (%):")
    print(f"  Logit:      {dp_logit}")
    print(f"  Linear:     {dp_linear}")
    print(f"  Log-linear: {dp_loglinear}")
    print()

    # =====================================================================
    # Welfare Analysis
    # =====================================================================
    # Pre-merger welfare
    cs_pre_logit = cs_logit(prices_obs, cal_logit['alpha'], cal_logit['xi'], M)
    ps_pre_logit = producer_surplus(prices_obs, shares_obs * M, mc_logit)

    q_pre_linear = shares_linear(prices_obs, cal_linear['a'], cal_linear['B'])
    cs_pre_linear = cs_linear(prices_obs, cal_linear['a'], cal_linear['B'],
                              np.linalg.solve(cal_linear['B'], cal_linear['a']))
    ps_pre_linear = producer_surplus(prices_obs, q_pre_linear, mc_linear)

    q_pre_loglinear = shares_loglinear(prices_obs, cal_loglinear['a_ll'], cal_loglinear['E'])
    p_high_ll = prices_obs * 5.0  # High price for integration bound
    cs_pre_loglinear = cs_loglinear(prices_obs, p_high_ll, cal_loglinear['a_ll'], cal_loglinear['E'])
    ps_pre_loglinear = producer_surplus(prices_obs, q_pre_loglinear, mc_loglinear)

    # Post-merger welfare
    cs_post_logit = cs_logit(p_post_logit, cal_logit['alpha'], cal_logit['xi'], M)
    ps_post_logit = producer_surplus(p_post_logit, s_post_logit * M, mc_logit)

    cs_post_linear = cs_linear(p_post_linear, cal_linear['a'], cal_linear['B'],
                               np.linalg.solve(cal_linear['B'], cal_linear['a']))
    ps_post_linear = producer_surplus(p_post_linear, q_post_linear, mc_linear)

    cs_post_loglinear = cs_loglinear(p_post_loglinear, p_high_ll, cal_loglinear['a_ll'], cal_loglinear['E'])
    ps_post_loglinear = producer_surplus(p_post_loglinear, q_post_loglinear, mc_loglinear)

    # Changes
    dCS = {
        "Logit": cs_post_logit - cs_pre_logit,
        "Linear": cs_post_linear - cs_pre_linear,
        "Log-linear": cs_post_loglinear - cs_pre_loglinear,
    }
    dPS = {
        "Logit": ps_post_logit - ps_pre_logit,
        "Linear": ps_post_linear - ps_pre_linear,
        "Log-linear": ps_post_loglinear - ps_pre_loglinear,
    }
    dW = {k: dCS[k] + dPS[k] for k in dCS}

    print("Welfare changes:")
    for model in ["Logit", "Linear", "Log-linear"]:
        print(f"  {model:12s}: dCS = {dCS[model]:+.4f}, dPS = {dPS[model]:+.4f}, dW = {dW[model]:+.4f}")
    print()

    # =====================================================================
    # Generate Report
    # =====================================================================
    setup_style()

    report = ModelReport(
        "Merger Pricing in Differentiated-Products Markets",
        "GUPPI screens and Bertrand-Nash counterfactuals with calibrated substitution.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A merger between close substitutes changes pricing incentives even if marginal costs "
        "do not move. Before the merger, Firm 1 treats sales lost to Firm 2 as business lost "
        "to a rival. After the merger, some of those consumers stay inside the same portfolio. "
        "That internalized diversion is the unilateral price effect an antitrust merger "
        "simulation tries to measure.\n\n"
        "The missing object is the substitution matrix. Shares, prices, and margins describe "
        "the observed market, but they do not directly reveal how many consumers leave Product "
        "1 for Product 3 when Product 1 becomes more expensive. This tutorial fixes a "
        "six-product market, calibrates logit, linear, and log-linear demand to the same "
        "observables, changes only ownership, and solves the post-merger Bertrand-Nash pricing "
        "conditions.\n\n"
        "GUPPI and CMCR summarize local upward pricing pressure at observed prices. The full "
        "counterfactual asks what happens after every firm reacts and after claimed cost "
        "efficiencies shift marginal costs. The logit-only [Bertrand pricing](../bertrand-logit-demand/) "
        "tutorial isolates the ownership matrix in one demand model. The [BLP random coefficients](../blp-random-coefficients/) "
        "tutorial shows how richer substitution patterns can be estimated. This page focuses "
        "on the gap between first-order screens and a solved post-merger equilibrium."
    )

    report.add_equations(r"""
The target object is the counterfactual price vector after ownership changes.
There are $J$ inside products. Product $j$ has price $p_j$, marginal cost
$c_j$, quantity or share $q_j(p)$, and owner $f(j)$. The ownership matrix is

$$
\Omega_{jk}=\mathbf 1\{f(j)=f(k)\}.
$$

For a multi-product Bertrand firm, the pricing equation is

$$
0=q_j(p)+\sum_{k=1}^J
\Omega_{jk}(p_k-c_k)\frac{\partial q_k(p)}{\partial p_j},
\qquad j=1,\ldots,J .
$$

With $\Delta_{kj}(p)=\partial q_k(p)/\partial p_j$, the vector equation is

$$
q(p)+(\Omega\circ \Delta(p)') (p-c)=0.
$$

The three demand systems are calibrated to the same observed market:

$$
s_j^{L}(p)=
\frac{\exp(\xi_j+\alpha p_j)}
{1+\sum_{\ell=1}^J \exp(\xi_\ell+\alpha p_\ell)},
\qquad \alpha<0,
$$

$$
q_j^{A}(p)=a_j-\sum_{k=1}^J B_{jk}p_k,
$$

and

$$
\log q_j^{E}(p)=a_j^E+\sum_{k=1}^J E_{jk}\log p_k .
$$

The local diversion ratio from product $j$ to product $k$ is

$$
D_{j\to k}=
-\frac{\partial q_k(p)/\partial p_j}
{\partial q_j(p)/\partial p_j}, \qquad j\neq k.
$$

For products that become newly co-owned after the merger,

$$
UPP_j=\sum_{k:\Omega^{post}_{jk}=1,\ \Omega^{pre}_{jk}=0}
D_{j\to k}(p_k-c_k),
$$

with

$$
GUPPI_j=\frac{UPP_j}{p_j},
\qquad
CMCR_j=\frac{UPP_j}{c_j}.
$$

GUPPI is a first-order screen for upward pricing pressure. CMCR reports the
product-level marginal-cost reduction that would offset that pressure at the
observed price vector. The simulation then solves the full pricing system under
post-merger ownership.
""")

    report.add_model_setup(
        "| Parameter | Value | Description |\n"
        "|-----------|-------|-------------|\n"
        f"| Products $J$ | {J} | 3 firms, 2 products each |\n"
        f"| Shares | {fmt_vector(shares_obs)} | Pre-merger inside shares |\n"
        f"| Prices | {fmt_vector(prices_obs)} | Pre-merger prices |\n"
        f"| Margins | {fmt_vector(margins_obs)} | Price-cost margins |\n"
        f"| Outside share | {1-np.sum(shares_obs):.2f} | Outside option in the logit demand system |\n"
        f"| $\\alpha$ (logit) | {cal_logit['alpha']:.4f} | Calibrated price coefficient |\n"
        "| Linear cross-slope ratio | 0.10 | Cross-slope relative to geometric mean own-slope |\n"
        "| Log-linear cross elasticity | 0.15 | Maintained symmetric cross-price elasticity |\n"
        "| Merger | Firm 1 buys Firm 2 | Products 1-4 move under common ownership |\n"
        "| Benchmark | Post-merger Bertrand-Nash FOC | Equilibrium used to judge first-order screens |"
    )

    report.add_solution_method(
        "Calibration and simulation do different jobs. Calibration makes each demand system "
        "pass through the observed market. Simulation keeps those demand primitives fixed, "
        "replaces the pre-merger ownership matrix with the post-merger one, and searches for "
        "a new Bertrand-Nash price vector.\n\n"
        "```text\n"
        "Algorithm: calibrated merger simulation\n"
        "Input: observed shares q, prices p, margins m, pre- and post-merger owners f(j)\n"
        "Output: screening metrics, post-merger prices, and welfare changes\n"
        "Build Omega_pre and Omega_post from owner labels\n"
        "for each demand system d in {logit, linear, log-linear}:\n"
        "    choose demand parameters so q_d(p_obs) matches observed shares\n"
        "    recover marginal costs c_d from the pre-merger pricing FOC\n"
        "    evaluate Delta_d(p_obs) and diversion ratios D_d\n"
        "    compute UPP, GUPPI, and CMCR for newly co-owned products\n"
        "    solve q_d(p) + (Omega_post .* Delta_d(p)') (p - c_d) = 0\n"
        "    compare solved price changes with the first-order screens\n"
        "    compute changes in consumer surplus, producer surplus, and total surplus\n"
        "for a grid of merger efficiencies:\n"
        "    reduce costs on the merging products, re-solve the post-merger FOC,\n"
        "    and interpolate the cost reduction where average merged-product prices stop rising\n"
        "```\n\n"
        f"The residual check confirms that each calibration reproduces the observed pricing "
        f"conditions "
        f"(logit {np.max(np.abs(foc_check_logit)):.1e}, "
        f"linear {np.max(np.abs(foc_check_linear)):.1e}, "
        f"log-linear {np.max(np.abs(foc_check_loglinear)):.1e}). "
        "The results below compare first-order screens with the solved post-merger "
        "equilibrium, rather than comparing three separately estimated demand models."
    )

    # -----------------------------------------------------------------
    # Figure 1: Pre vs post-merger prices by demand model
    # -----------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    demand_labels = ["Logit", "Linear", "Log-linear"]
    post_prices_all = [p_post_logit, p_post_linear, p_post_loglinear]
    colors_pre_post = ["steelblue", "coral"]

    x = np.arange(J)
    bar_w = 0.35
    for idx, (ax, label, p_post) in enumerate(zip(axes1, demand_labels, post_prices_all)):
        ax.bar(x - bar_w / 2, prices_obs, bar_w, label="Pre-merger", color=colors_pre_post[0])
        ax.bar(x + bar_w / 2, p_post, bar_w, label="Post-merger", color=colors_pre_post[1])
        ax.set_xlabel("Product")
        ax.set_title(f"{label} Demand")
        ax.set_xticks(x)
        ax.set_xticklabels(product_names, fontsize=8, rotation=45)
        ax.legend(fontsize=8)
        # Mark merging products
        for j in range(4):
            pct = (p_post[j] - prices_obs[j]) / prices_obs[j] * 100
            ax.annotate(f"+{pct:.1f}%", xy=(j + bar_w / 2, p_post[j]),
                        fontsize=7, ha="center", va="bottom", color="red")
    axes1[0].set_ylabel("Price")
    fig1.suptitle("Pre- vs Post-Merger Prices by Demand Model", fontsize=14, y=1.02)
    fig1.tight_layout()
    report.add_figure(
        "figures/price-comparison.png",
        "Pre- vs post-merger prices across three demand systems. Merging products (1-4) "
        "see larger price increases, with magnitudes shaped by the calibrated substitution pattern.",
        fig1,
        description="Products 1-4 move inside the same portfolio after the merger, so the "
        "merged firm internalizes diversion among them. The logit and log-linear systems give "
        "roughly double-digit average increases for those products. The linear system is more "
        "muted under this cross-slope calibration.",
    )

    # -----------------------------------------------------------------
    # Figure 2: Welfare decomposition (CS, PS, total)
    # -----------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    models = ["Logit", "Linear", "Log-linear"]
    x2 = np.arange(len(models))
    bar_w2 = 0.25
    cs_vals = [dCS[m] for m in models]
    ps_vals = [dPS[m] for m in models]
    w_vals = [dW[m] for m in models]

    ax2.bar(x2 - bar_w2, cs_vals, bar_w2, label="$\\Delta$ CS", color="steelblue")
    ax2.bar(x2, ps_vals, bar_w2, label="$\\Delta$ PS", color="coral")
    ax2.bar(x2 + bar_w2, w_vals, bar_w2, label="$\\Delta$ W (total)", color="seagreen")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax2.set_xlabel("Demand Model")
    ax2.set_ylabel("Welfare Change")
    ax2.set_title("Welfare Decomposition: Consumer, Producer, and Total Surplus")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(models)
    ax2.legend()
    report.add_figure(
        "figures/welfare-decomposition.png",
        "Welfare decomposition across demand systems: consumers lose, producers may gain, "
        "and the net effect depends on the demand model.",
        fig2,
        description="The welfare bars separate consumer surplus, producer surplus, and their "
        "sum. Consumers lose in every demand system here. Producer surplus rises, but the gain "
        "does not offset the consumer loss, so total surplus falls under this calibration.",
    )

    # -----------------------------------------------------------------
    # Figure 3: Screening metrics vs solved equilibrium
    # -----------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
    model_colors = ["steelblue", "coral", "seagreen"]
    dp_by_model = {"Logit": dp_logit, "Linear": dp_linear, "Log-linear": dp_loglinear}
    guppi_by_model = {"Logit": guppi_logit, "Linear": guppi_linear, "Log-linear": guppi_loglinear}
    upp_by_model = {"Logit": upp_logit, "Linear": upp_linear, "Log-linear": upp_loglinear}

    x3 = np.arange(len(demand_labels))
    bar_w3 = 0.35
    avg_guppi = [np.mean(guppi_by_model[label][:4]) * 100 for label in demand_labels]
    avg_actual = [np.mean(dp_by_model[label][:4]) for label in demand_labels]
    ax3a.bar(x3 - bar_w3 / 2, avg_guppi, bar_w3, label="GUPPI screen", color="gray")
    ax3a.bar(x3 + bar_w3 / 2, avg_actual, bar_w3, label="Solved equilibrium", color="seagreen")
    ax3a.set_xlabel("Demand Model")
    ax3a.set_ylabel("Average Merging-Product Effect (%)")
    ax3a.set_title("Screen vs Solved Price Increase")
    ax3a.set_xticks(x3)
    ax3a.set_xticklabels(demand_labels)
    ax3a.legend(fontsize=9)
    ax3a.axhline(0, color="black", linewidth=0.8)

    xprod = np.arange(4)
    bar_w_upp = 0.25
    for idx, label in enumerate(demand_labels):
        ax3b.bar(
            xprod + (idx - 1) * bar_w_upp,
            upp_by_model[label][:4],
            bar_w_upp,
            label=label,
            color=model_colors[idx],
        )
    ax3b.set_xlabel("Product")
    ax3b.set_ylabel("UPP")
    ax3b.set_title("UPP for Newly Co-Owned Products")
    ax3b.set_xticks(xprod)
    ax3b.set_xticklabels(product_names[:4], fontsize=9)
    ax3b.legend(fontsize=9)
    ax3b.axhline(0, color="black", linewidth=0.8)

    fig3.tight_layout()
    report.add_figure(
        "figures/upp-guppi.png",
        "GUPPI screen versus solved equilibrium price effects, with product-level UPP for "
        "the products that become newly co-owned.",
        fig3,
        description="GUPPI uses observed margins and diversion before the counterfactual "
        "prices are solved. The left panel treats the solved post-merger price increase as "
        "the benchmark. The gap comes from pass-through, demand curvature, and rival price "
        "responses.",
    )

    # -----------------------------------------------------------------
    # Figure 4: Price effects vs efficiency gains frontier
    # -----------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(9, 6))
    efficiency_levels = np.linspace(0.0, 0.60, 121)  # 0% to 60% cost reduction

    mc_by_model = {"Logit": mc_logit, "Linear": mc_linear, "Log-linear": mc_loglinear}
    post_price_by_model = {"Logit": p_post_logit, "Linear": p_post_linear, "Log-linear": p_post_loglinear}
    break_even_efficiency = {}
    for idx, (label, color) in enumerate(zip(demand_labels, model_colors)):
        avg_price_changes = []
        mc_base = mc_by_model[label]
        p_warm = post_price_by_model[label].copy()  # Warmstart from post-merger solution
        for eff in efficiency_levels:
            mc_eff = mc_base.copy()
            mc_eff[:4] = mc_base[:4] * (1.0 - eff)  # Cost reduction only for merging firms

            if label == "Logit":
                p_eff = scipy.optimize.fsolve(
                    foc_logit, x0=p_warm,
                    args=(mc_eff, cal_logit['alpha'], cal_logit['xi'], omega_post),
                    full_output=False
                )
            elif label == "Linear":
                p_eff = scipy.optimize.fsolve(
                    foc_linear, x0=p_warm,
                    args=(mc_eff, cal_linear['a'], cal_linear['B'], omega_post),
                    full_output=False
                )
            else:
                logp_eff = scipy.optimize.fsolve(
                    foc_loglinear_logp, x0=np.log(np.maximum(p_warm, 0.01)),
                    args=(mc_eff, cal_loglinear['a_ll'], cal_loglinear['E'], omega_post),
                    full_output=False
                )
                p_eff = np.exp(logp_eff)
            p_warm = p_eff.copy()  # Use this solution as warmstart for next level
            avg_dp = np.mean((p_eff[:4] - prices_obs[:4]) / prices_obs[:4]) * 100
            avg_price_changes.append(avg_dp)

        efficiency_pct = efficiency_levels * 100
        break_even_efficiency[label] = first_zero_crossing(efficiency_pct, avg_price_changes)
        ax4.plot(efficiency_pct, avg_price_changes, "o-", label=label,
                 color=color, markersize=4)
        if np.isfinite(break_even_efficiency[label]):
            ax4.scatter(
                [break_even_efficiency[label]], [0.0],
                color=color, edgecolor="black", linewidth=0.6, zorder=5,
            )

    ax4.axhline(0, color="black", linewidth=1.0, linestyle="--")
    ax4.set_xlabel("Marginal Cost Reduction for Merging Firms (%)")
    ax4.set_ylabel("Avg Price Change for Merging Products (%)")
    ax4.set_title("Price Effects vs Efficiency Gains Frontier")
    ax4.legend()
    ax4.annotate("Price-increasing\nmerger", xy=(2, 2), fontsize=9, color="red",
                 ha="left", va="bottom")
    ax4.annotate("Consumer-beneficial\nmerger", xy=(20, -2), fontsize=9, color="green",
                 ha="right", va="top")
    report.add_figure(
        "figures/efficiency-frontier.png",
        "How much marginal cost reduction is needed to offset the merger price increase? "
        "The break-even point differs substantially across demand models.",
        fig4,
        description="The efficiency curve lowers marginal costs for products 1-4 and re-solves "
        "the post-merger pricing problem. The zero markers are interpolated from a fine "
        "efficiency grid, so they approximate the solved-equilibrium break-even point. Below "
        "the zero line, efficiencies are large enough to reverse the average price increase "
        "on the merged products.",
    )

    # -----------------------------------------------------------------
    # Table: Merger effects comparison
    # -----------------------------------------------------------------
    table_data = {
        "Demand Model": [],
        "Avg Actual Price Inc. (%)": [],
        "Max Price Change (%)": [],
        "Avg GUPPI Screen (%)": [],
        "Screen Gap (pp)": [],
        "Avg CMCR Screen (%)": [],
        "Break-even Eff. (%)": [],
        "Delta CS": [],
        "Delta PS": [],
        "Delta W": [],
        "Post FOC Residual": [],
    }
    post_residuals = {
        "Logit": np.max(np.abs(foc_post_logit)),
        "Linear": np.max(np.abs(foc_post_linear)),
        "Log-linear": np.max(np.abs(foc_post_loglinear)),
    }

    for label, p_post, guppi_vals, cmcr_vals in [
        ("Logit", p_post_logit, guppi_logit, cmcr_logit),
        ("Linear", p_post_linear, guppi_linear, cmcr_linear),
        ("Log-linear", p_post_loglinear, guppi_loglinear, cmcr_loglinear),
    ]:
        dp = (p_post - prices_obs) / prices_obs * 100
        avg_dp = np.mean(dp[:4])
        avg_guppi_val = np.mean(guppi_vals[:4]) * 100
        table_data["Demand Model"].append(label)
        table_data["Avg Actual Price Inc. (%)"].append(round(avg_dp, 2))
        table_data["Max Price Change (%)"].append(round(np.max(dp[:4]), 2))
        table_data["Avg GUPPI Screen (%)"].append(round(avg_guppi_val, 2))
        table_data["Screen Gap (pp)"].append(round(avg_dp - avg_guppi_val, 2))
        table_data["Avg CMCR Screen (%)"].append(round(np.mean(cmcr_vals[:4]) * 100, 2))
        table_data["Break-even Eff. (%)"].append(round(break_even_efficiency[label], 2))
        table_data["Delta CS"].append(round(dCS[label], 4))
        table_data["Delta PS"].append(round(dPS[label], 4))
        table_data["Delta W"].append(round(dW[label], 4))
        table_data["Post FOC Residual"].append(f"{post_residuals[label]:.1e}")

    df = pd.DataFrame(table_data)
    report.add_table("tables/merger-effects.csv", "Merger Price Effects and Screens", df,
        description="The table puts local screens and solved counterfactuals side by side. "
        "Average actual price increases come from the post-merger FOC solution. GUPPI and "
        "CMCR are local screens. The break-even efficiency column comes from re-solving the "
        "pricing equilibrium on a finer cost-reduction grid.")

    # -----------------------------------------------------------------
    # Takeaway
    # -----------------------------------------------------------------
    report.add_takeaway(
        "Merger simulation turns a change in control into an equilibrium price calculation. "
        "The ownership matrix is easy to change. The economic content sits in substitution "
        "and cost pass-through, so different calibrated demand systems can give different "
        "counterfactuals in the same observed market.\n\n"
        "UPP, GUPPI, and CMCR are useful triage tools. They point to products with strong "
        "internalized diversion, while the FOC solve accounts for every firm's reaction, "
        "demand curvature, and cost-efficiency claims. In this calibration, all three systems "
        "raise prices and lower consumer surplus, but they disagree on magnitudes and on the "
        "efficiency needed to offset the merger."
    )

    report.add_references([
        "Werden, G. and Froeb, L. (1994). \"The Effects of Mergers in Differentiated Products "
        "Industries: Logit Demand and Merger Policy.\" *Journal of Law, Economics, & Organization*, 10(2).",
        "Farrell, J. and Shapiro, C. (2010). \"Antitrust Evaluation of Horizontal Mergers: "
        "An Economic Alternative to Market Definition.\" *The B.E. Journal of Theoretical Economics*, 10(1).",
        "Werden, G. (1996). \"A Robust Test for Consumer Welfare Enhancing Mergers Among "
        "Sellers of Differentiated Products.\" *Journal of Industrial Economics*, 44(4).",
        "Nevo, A. (2000). \"Mergers with Differentiated Products: The Case of the Ready-to-Eat "
        "Cereal Industry.\" *RAND Journal of Economics*, 31(3).",
        "Berry, S., Levinsohn, J., and Pakes, A. (1995). \"Automobile Prices in Market "
        "Equilibrium.\" *Econometrica*, 63(4).",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
