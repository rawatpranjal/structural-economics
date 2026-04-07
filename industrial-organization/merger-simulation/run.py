#!/usr/bin/env python3
"""Merger Simulation with Multiple Demand Systems.

Compares merger price effects, welfare, and screening metrics (UPP, GUPPI, CMCR)
across logit, linear, and log-linear demand specifications. The choice of demand
model matters enormously for merger analysis: logit overstates substitution to
the outside good, linear demand understates it, and different functional forms
lead to very different policy conclusions.

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
    """Calibrate logit demand from observed shares, prices, and margins."""
    mc = prices_obs * (1.0 - margins_obs)
    omega = ownership_matrix(p2f)

    # Recover alpha from FOC for product 0:
    # s_0 + alpha * s_0 * (1-s_0) * (p_0 - mc_0) - alpha * sum_{k in F_0} s_0*s_k*(p_k-mc_k) = 0
    # For single-product firms: s_0 + alpha * s_0 * (1-s_0) * (p_0-mc_0) = 0
    # alpha = -1 / ((1-s_0) * (p_0-mc_0))
    # For multi-product firms, use the full FOC.
    s0 = shares_obs[0]
    markup0 = prices_obs[0] - mc[0]

    # Products owned by firm of product 0
    firm0 = p2f[0]
    same_firm = (p2f == firm0)
    # alpha from margin of product 0 (accounting for portfolio)
    denom = s0 * (1.0 - s0) * markup0
    for k in range(len(p2f)):
        if k != 0 and same_firm[k]:
            denom -= s0 * shares_obs[k] * (prices_obs[k] - mc[k])
    alpha = -s0 / denom

    # Mean valuations xi
    s0_total = np.sum(shares_obs)
    xi = np.log(shares_obs / (1.0 - s0_total)) - alpha * prices_obs

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

    product_names = [f"Prod {j+1}" for j in range(J)]
    firm_names_pre = ["Firm 1", "Firm 1", "Firm 2", "Firm 2", "Firm 3", "Firm 3"]

    # Post-merger: Firm 1 acquires Firm 2
    p2f_post = np.array([1, 1, 1, 1, 3, 3])

    print("=" * 70)
    print("MERGER SIMULATION: Multi-Demand-System Comparison")
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
    cal_loglinear = calibrate_loglinear(shares_obs, prices_obs, margins_obs, p2f_pre, cross_elas=0.3)

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
    # Screening metrics: UPP, GUPPI, CMCR
    # =====================================================================
    mc = cal_logit['mc']  # Same mc across all models

    upp_logit = compute_upp(div_logit, prices_obs, mc, p2f_pre, p2f_post)
    upp_linear = compute_upp(div_linear, prices_obs, mc, p2f_pre, p2f_post)
    upp_loglinear = compute_upp(div_loglinear, prices_obs, mc, p2f_pre, p2f_post)

    guppi_logit = compute_guppi(div_logit, prices_obs, mc, p2f_pre, p2f_post)
    guppi_linear = compute_guppi(div_linear, prices_obs, mc, p2f_pre, p2f_post)
    guppi_loglinear = compute_guppi(div_loglinear, prices_obs, mc, p2f_pre, p2f_post)

    cmcr_logit = compute_cmcr(div_logit, prices_obs, mc, p2f_pre, p2f_post)
    cmcr_linear = compute_cmcr(div_linear, prices_obs, mc, p2f_pre, p2f_post)
    cmcr_loglinear = compute_cmcr(div_loglinear, prices_obs, mc, p2f_pre, p2f_post)

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

    # Log-linear
    p_post_loglinear = scipy.optimize.fsolve(
        foc_loglinear, x0=prices_obs * 1.05,
        args=(cal_loglinear['mc'], cal_loglinear['a_ll'], cal_loglinear['E'], omega_post),
        full_output=False
    )
    q_post_loglinear = shares_loglinear(p_post_loglinear, cal_loglinear['a_ll'], cal_loglinear['E'])

    print(f"  Logit post-merger prices:      {p_post_logit}")
    print(f"  Linear post-merger prices:     {p_post_linear}")
    print(f"  Log-linear post-merger prices: {p_post_loglinear}")
    print()

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
    ps_pre_logit = producer_surplus(prices_obs, shares_obs * M, mc)

    q_pre_linear = shares_linear(prices_obs, cal_linear['a'], cal_linear['B'])
    cs_pre_linear = cs_linear(prices_obs, cal_linear['a'], cal_linear['B'],
                              np.linalg.solve(cal_linear['B'], cal_linear['a']))
    ps_pre_linear = producer_surplus(prices_obs, q_pre_linear, mc)

    q_pre_loglinear = shares_loglinear(prices_obs, cal_loglinear['a_ll'], cal_loglinear['E'])
    p_high_ll = prices_obs * 5.0  # High price for integration bound
    cs_pre_loglinear = cs_loglinear(prices_obs, p_high_ll, cal_loglinear['a_ll'], cal_loglinear['E'])
    ps_pre_loglinear = producer_surplus(prices_obs, q_pre_loglinear, mc)

    # Post-merger welfare
    cs_post_logit = cs_logit(p_post_logit, cal_logit['alpha'], cal_logit['xi'], M)
    ps_post_logit = producer_surplus(p_post_logit, s_post_logit * M, mc)

    cs_post_linear = cs_linear(p_post_linear, cal_linear['a'], cal_linear['B'],
                               np.linalg.solve(cal_linear['B'], cal_linear['a']))
    ps_post_linear = producer_surplus(p_post_linear, q_post_linear, mc)

    cs_post_loglinear = cs_loglinear(p_post_loglinear, p_high_ll, cal_loglinear['a_ll'], cal_loglinear['E'])
    ps_post_loglinear = producer_surplus(p_post_loglinear, q_post_loglinear, mc)

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
        "Merger Simulation with Multiple Demand Systems",
        "Comparing merger effects across logit, linear, and log-linear demand to show how "
        "functional form assumptions drive antitrust policy conclusions.",
    )

    report.add_overview(
        "Horizontal merger analysis in differentiated product markets hinges on the assumed "
        "demand system. This model calibrates three demand specifications --- logit, linear, "
        "and log-linear --- to identical pre-merger data, then simulates the same merger "
        "(Firm 1 acquires Firm 2) under each. The results diverge, illustrating a central "
        "challenge in structural merger analysis.\n\n"
        "**Why demand form matters:**\n"
        "- **Logit** imposes the IIA property: substitution to the outside good is proportional "
        "to share, which overstates escape to non-purchase and can understate price effects "
        "among close substitutes.\n"
        "- **Linear demand** has bounded quantities and a choke price, implying substitution "
        "patterns that differ qualitatively from discrete choice.\n"
        "- **Log-linear (constant elasticity)** demand has no choke price --- demand never "
        "reaches zero --- and often predicts the largest price increases because margins are "
        "sensitive to elasticity.\n\n"
        "We also compute standard antitrust screening metrics: UPP (Upward Pricing Pressure), "
        "GUPPI (Gross Upward Pricing Pressure Index), and CMCR (Compensating Marginal Cost "
        "Reduction), all of which vary across demand systems."
    )

    report.add_equations(r"""
**Bertrand-Nash FOC (general):**
$$q_j + \sum_{k \in \mathcal{F}_f} \frac{\partial q_k}{\partial p_j} (p_k - c_k) = 0 \quad \forall j \in \mathcal{F}_f$$

In matrix form: $\mathbf{q} + (\Omega \circ \mathbf{J}') (\mathbf{p} - \mathbf{c}) = 0$ where $\mathbf{J} = \partial \mathbf{q} / \partial \mathbf{p}'$.

**Logit:** $s_j = \frac{\exp(\xi_j + \alpha p_j)}{1 + \sum_k \exp(\xi_k + \alpha p_k)}$

**Linear:** $q_j = a_j - \sum_k B_{jk} p_k$

**Log-linear:** $\ln q_j = a_j + \sum_k E_{jk} \ln p_k$

**Diversion ratio:** $D_{j \to k} = -\frac{\partial q_k / \partial p_j}{\partial q_j / \partial p_j}$

**UPP:** $\text{UPP}_j = \sum_{k \text{ newly co-owned}} D_{j \to k} \cdot (p_k - c_k)$

**GUPPI:** $\text{GUPPI}_j = \text{UPP}_j / p_j$

**CMCR:** $\text{CMCR}_j = \text{UPP}_j / c_j$ --- marginal cost reduction needed to offset pricing pressure.
""")

    report.add_model_setup(
        "| Parameter | Value | Description |\n"
        "|-----------|-------|-------------|\n"
        f"| Products $J$ | {J} | 3 firms, 2 products each |\n"
        f"| Shares | {list(shares_obs)} | Pre-merger market shares |\n"
        f"| Prices | {list(prices_obs)} | Pre-merger prices |\n"
        f"| Margins | {list(margins_obs)} | Price-cost margins |\n"
        f"| Outside share | {1-np.sum(shares_obs):.2f} | Logit outside good |\n"
        f"| $\\alpha$ (logit) | {cal_logit['alpha']:.4f} | Calibrated price coefficient |\n"
        "| Cross-price ratio (linear) | 0.10 | Cross-slope / geometric mean of own-slopes |\n"
        "| Cross elasticity (log-linear) | 0.30 | Symmetric cross-price elasticities |\n"
        "| Merger | Firm 1 + Firm 2 | Products 1-4 under common ownership |"
    )

    report.add_solution_method(
        "**Step 1: Calibrate** each demand system from the same observed data (shares, prices, "
        "margins). The FOC is inverted to recover marginal costs, and demand parameters are "
        "chosen to match observed equilibrium.\n\n"
        f"**Step 2: Verify** FOC residuals at pre-merger prices "
        f"(logit: {np.max(np.abs(foc_check_logit)):.1e}, "
        f"linear: {np.max(np.abs(foc_check_linear)):.1e}, "
        f"log-linear: {np.max(np.abs(foc_check_loglinear)):.1e}).\n\n"
        "**Step 3: Screen** using UPP, GUPPI, and CMCR --- first-order approximations to "
        "merger harm that do not require solving the full post-merger equilibrium.\n\n"
        "**Step 4: Simulate** by changing the ownership matrix $\\Omega$ and solving the new "
        "Bertrand-Nash equilibrium via `scipy.optimize.fsolve`.\n\n"
        "**Step 5: Evaluate** welfare changes: consumer surplus (CS), producer surplus (PS), "
        "and total welfare ($W = CS + PS$)."
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
        "see larger price increases; the magnitude depends heavily on the demand model.",
        fig1,
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
    )

    # -----------------------------------------------------------------
    # Figure 3: UPP and GUPPI by product
    # -----------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
    x3 = np.arange(J)
    bar_w3 = 0.25
    model_colors = ["steelblue", "coral", "seagreen"]

    # UPP
    for idx, (label, upp_vals) in enumerate(zip(demand_labels,
            [upp_logit, upp_linear, upp_loglinear])):
        ax3a.bar(x3 + idx * bar_w3, upp_vals, bar_w3, label=label, color=model_colors[idx])
    ax3a.set_xlabel("Product")
    ax3a.set_ylabel("UPP")
    ax3a.set_title("Upward Pricing Pressure by Product")
    ax3a.set_xticks(x3 + bar_w3)
    ax3a.set_xticklabels(product_names, fontsize=8, rotation=45)
    ax3a.legend(fontsize=9)
    ax3a.axhline(0, color="black", linewidth=0.8)

    # GUPPI
    for idx, (label, guppi_vals) in enumerate(zip(demand_labels,
            [guppi_logit, guppi_linear, guppi_loglinear])):
        ax3b.bar(x3 + idx * bar_w3, guppi_vals * 100, bar_w3, label=label, color=model_colors[idx])
    ax3b.set_xlabel("Product")
    ax3b.set_ylabel("GUPPI (%)")
    ax3b.set_title("Gross Upward Pricing Pressure Index by Product")
    ax3b.set_xticks(x3 + bar_w3)
    ax3b.set_xticklabels(product_names, fontsize=8, rotation=45)
    ax3b.legend(fontsize=9)
    ax3b.axhline(0, color="black", linewidth=0.8)

    fig3.tight_layout()
    report.add_figure(
        "figures/upp-guppi.png",
        "UPP and GUPPI by product and demand model. Only merging products (1-4) have "
        "positive values; non-merging products have zero UPP by construction.",
        fig3,
    )

    # -----------------------------------------------------------------
    # Figure 4: Price effects vs efficiency gains frontier
    # -----------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(9, 6))
    efficiency_levels = np.linspace(0.0, 0.25, 15)  # 0% to 25% cost reduction

    for idx, (label, color) in enumerate(zip(demand_labels, model_colors)):
        avg_price_changes = []
        for eff in efficiency_levels:
            mc_eff = mc.copy()
            mc_eff[:4] = mc[:4] * (1.0 - eff)  # Cost reduction only for merging firms

            if label == "Logit":
                p_eff = scipy.optimize.fsolve(
                    foc_logit, x0=prices_obs * 1.02,
                    args=(mc_eff, cal_logit['alpha'], cal_logit['xi'], omega_post),
                    full_output=False
                )
            elif label == "Linear":
                p_eff = scipy.optimize.fsolve(
                    foc_linear, x0=prices_obs * 1.02,
                    args=(mc_eff, cal_linear['a'], cal_linear['B'], omega_post),
                    full_output=False
                )
            else:
                p_eff = scipy.optimize.fsolve(
                    foc_loglinear, x0=prices_obs * 1.02,
                    args=(mc_eff, cal_loglinear['a_ll'], cal_loglinear['E'], omega_post),
                    full_output=False
                )
            avg_dp = np.mean((p_eff[:4] - prices_obs[:4]) / prices_obs[:4]) * 100
            avg_price_changes.append(avg_dp)

        ax4.plot(efficiency_levels * 100, avg_price_changes, "o-", label=label,
                 color=color, markersize=4)

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
    )

    # -----------------------------------------------------------------
    # Table: Merger effects comparison
    # -----------------------------------------------------------------
    table_data = {
        "Demand Model": [],
        "Avg Price Change (%)": [],
        "Max Price Change (%)": [],
        "Delta CS": [],
        "Delta PS": [],
        "Delta W": [],
        "Avg GUPPI (%)": [],
        "Avg CMCR (%)": [],
    }

    for label, p_post, guppi_vals, cmcr_vals in [
        ("Logit", p_post_logit, guppi_logit, cmcr_logit),
        ("Linear", p_post_linear, guppi_linear, cmcr_linear),
        ("Log-linear", p_post_loglinear, guppi_loglinear, cmcr_loglinear),
    ]:
        dp = (p_post - prices_obs) / prices_obs * 100
        table_data["Demand Model"].append(label)
        table_data["Avg Price Change (%)"].append(f"{np.mean(dp[:4]):.2f}")
        table_data["Max Price Change (%)"].append(f"{np.max(dp[:4]):.2f}")
        table_data["Delta CS"].append(f"{dCS[label]:+.4f}")
        table_data["Delta PS"].append(f"{dPS[label]:+.4f}")
        table_data["Delta W"].append(f"{dW[label]:+.4f}")
        table_data["Avg GUPPI (%)"].append(f"{np.mean(guppi_vals[:4]) * 100:.2f}")
        table_data["Avg CMCR (%)"].append(f"{np.mean(cmcr_vals[:4]) * 100:.2f}")

    df = pd.DataFrame(table_data)
    report.add_table("tables/merger-effects.csv", "Merger Effects Comparison Across Demand Models", df)

    # -----------------------------------------------------------------
    # Takeaway
    # -----------------------------------------------------------------
    report.add_takeaway(
        "The choice of demand functional form is not innocuous --- it is arguably the most "
        "consequential modeling decision in structural merger simulation.\n\n"
        "**Key insights:**\n"
        "- **Logit demand** imposes IIA: all products (including the outside good) absorb "
        "diverted sales in proportion to their shares. This typically understates harm from "
        "mergers between close substitutes because too much substitution 'escapes' to non-purchase.\n"
        "- **Linear demand** has finite choke prices and bounded substitution. Cross-price "
        "effects depend on the assumed cross-slope parameters, making results sensitive to "
        "calibration choices.\n"
        "- **Log-linear (constant elasticity) demand** has no choke price and can imply very "
        "large price effects, especially when own-price elasticities are low (high margins).\n"
        "- **UPP and GUPPI** are first-order approximations that avoid solving the full "
        "post-merger equilibrium. They are useful screens but cannot capture feedback effects "
        "(rivals' price responses, demand curvature).\n"
        "- **CMCR** translates merger harm into the language of efficiency: how large must cost "
        "synergies be to leave consumers no worse off? This is the standard the DOJ/FTC apply.\n"
        "- The **efficiency frontier** plot shows that the break-even cost reduction can differ "
        "by a factor of two or more across demand models --- a sobering reminder that policy "
        "conclusions are model-dependent.\n\n"
        "The practical lesson: robust merger analysis should present results under multiple "
        "demand specifications, not rely on a single functional form."
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
