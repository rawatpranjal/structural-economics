#!/usr/bin/env python3
"""Merger pricing with concentration screens and calibrated demand systems.

Three layers an antitrust analyst would actually use in sequence:

(a) Concentration screens. HHI, delta-HHI, and the effective-firms count
    work directly from the sales table. They are arithmetic, not pricing
    models.

(b) A four-product Bertrand-Nash baseline with logit demand. One observed
    margin pins down the price coefficient. Marginal costs come from the
    pre-merger pricing FOC. Ownership changes resolve the same FOC.

(c) A six-product extension with three demand systems (logit, linear,
    log-linear). UPP, GUPPI, CMCR are local screens; full counterfactual
    prices come from the post-merger FOC; welfare and an efficiency-gain
    frontier follow.

References: Werden and Froeb (1994), Farrell and Shapiro (2010),
U.S. Department of Justice and Federal Trade Commission (2023).
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
# Concentration screens (HHI, delta-HHI, effective firms)
# =========================================================================

def compute_hhi(shares: np.ndarray) -> float:
    """HHI on the standard 0-10000 scale: HHI = 10000 * sum_f s_f^2."""
    shares = np.asarray(shares, dtype=float)
    return float(np.sum(shares ** 2) * 10000.0)


def effective_firms(shares: np.ndarray) -> float:
    """Equivalent number of equal-sized firms implied by the HHI."""
    return 10000.0 / compute_hhi(shares)


def delta_hhi(s_a: float, s_b: float) -> float:
    """Increase in HHI from merging two firms with quantities held fixed."""
    return 2.0 * float(s_a) * float(s_b) * 10000.0


def classify_hhi(hhi: float) -> str:
    """2023 DOJ/FTC structural classification."""
    if hhi < 1000:
        return "Unconcentrated"
    if hhi <= 1800:
        return "Moderately Concentrated"
    return "Highly Concentrated"


def equal_shares(n: int) -> np.ndarray:
    """Symmetric shares for n equal-sized firms."""
    return np.ones(n) / n


# =========================================================================
# Ownership matrix (shared across both small and extended cases)
# =========================================================================

def ownership_matrix(p2f: np.ndarray) -> np.ndarray:
    """Ownership matrix: Omega[j,k] = 1 if products j and k share an owner."""
    p2f = np.asarray(p2f)
    return (p2f[:, None] == p2f[None, :]).astype(float)


# =========================================================================
# Demand System 1: Logit
# =========================================================================

def shares_logit(p: np.ndarray, alpha: float, xi: np.ndarray) -> np.ndarray:
    """Logit shares: s_j = exp(xi_j + alpha p_j) / (1 + sum exp(...))."""
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
    """Bertrand-Nash FOC: s + (Omega .* dsdp') (p - mc) = 0."""
    s = shares_logit(p, alpha, xi)
    dsdp = jacobian_logit(p, alpha, xi)
    return s + (omega * dsdp.T) @ (p - mc)


def cs_logit(p: np.ndarray, alpha: float, xi: np.ndarray, M: float) -> float:
    """Consumer surplus (logit): CS = -M/alpha * ln(1 + sum exp(xi + alpha p))."""
    v = np.exp(xi + alpha * p)
    return -M / alpha * np.log(1.0 + np.sum(v))


# =========================================================================
# Demand System 2: Linear
# =========================================================================

def shares_linear(p: np.ndarray, a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Linear demand: q = a - B p, returned as quantities (interpreted as shares of M=1)."""
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
    """Consumer surplus for linear demand via the quadratic form 0.5 q' B^{-1} q."""
    q = shares_linear(p, a, B)
    try:
        B_inv = np.linalg.inv(B)
        return 0.5 * q @ B_inv @ q
    except np.linalg.LinAlgError:
        return 0.5 * np.sum(q ** 2 / np.diag(B))


# =========================================================================
# Demand System 3: Log-Linear
# =========================================================================

def shares_loglinear(p: np.ndarray, a_ll: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Log-linear demand: log q_j = a_j + sum_k E_{jk} log p_k."""
    log_q = a_ll + E @ np.log(p)
    return np.exp(log_q)


def jacobian_loglinear(p: np.ndarray, a_ll: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Jacobian dq_j/dp_k = q_j * E_{jk} / p_k for log-linear demand."""
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
    """FOC in log-price space, keeps prices positive during root finding."""
    return foc_loglinear(np.exp(log_p), mc, a_ll, E, omega)


def cs_loglinear(p: np.ndarray, p_high: np.ndarray, a_ll: np.ndarray,
                 E: np.ndarray, n_steps: int = 200) -> float:
    """Consumer surplus for log-linear demand via numerical integration along p -> p_high."""
    cs = 0.0
    for i in range(n_steps):
        t_mid = (i + 0.5) / n_steps
        p_mid = p + t_mid * (p_high - p)
        q_mid = shares_loglinear(p_mid, a_ll, E)
        dp = (p_high - p) / n_steps
        cs += np.sum(q_mid * dp)
    return cs


# =========================================================================
# Calibration
# =========================================================================

def calibrate_logit_simple(shares_obs: np.ndarray, prices_obs: np.ndarray,
                           margin_first: float, p2f: np.ndarray) -> dict:
    """Calibrate logit from one observed margin.

    The single-product FOC for product 1 gives alpha = -1 / [(1 - s_1)(p_1 - c_1)].
    With c_1 = p_1 (1 - margin), alpha is pinned down. Marginal costs for the
    remaining products fall out of the full multi-product FOC.
    """
    omega = ownership_matrix(p2f)
    c1 = prices_obs[0] * (1.0 - margin_first)
    alpha = -1.0 / (1.0 - shares_obs[0]) / (prices_obs[0] - c1)
    s0 = 1.0 - np.sum(shares_obs)
    xi = np.log(shares_obs / s0) - alpha * prices_obs
    dsdp = jacobian_logit(prices_obs, alpha, xi)
    mc = prices_obs + np.linalg.solve(omega * dsdp.T, shares_obs)
    return {"alpha": alpha, "xi": xi, "mc": mc}


def calibrate_logit(shares_obs: np.ndarray, prices_obs: np.ndarray,
                    margins_obs: np.ndarray, p2f: np.ndarray) -> dict:
    """Calibrate logit demand from observed shares, prices, and a margin vector.

    Strategy: average each product's single-product alpha estimate to pin down
    a single price coefficient, then invert the full pricing FOC for marginal
    costs.
    """
    omega = ownership_matrix(p2f)
    s0_total = np.sum(shares_obs)
    # Single-product logit FOC: 0 = s_j + (p_j - c_j) * alpha * s_j * (1 - s_j),
    # with p_j - c_j = m_j * p_j, gives alpha = -1 / (m_j * p_j * (1 - s_j)).
    # The price factor is load-bearing whenever observed prices are not unity.
    alpha_estimates = -1.0 / (margins_obs * prices_obs * (1.0 - shares_obs))
    alpha = float(np.mean(alpha_estimates))
    xi = np.log(shares_obs / (1.0 - s0_total)) - alpha * prices_obs
    dsdp = jacobian_logit(prices_obs, alpha, xi)
    mc = prices_obs + np.linalg.solve(omega * dsdp.T, shares_obs)
    return {"alpha": alpha, "xi": xi, "mc": mc}


def calibrate_linear(shares_obs: np.ndarray, prices_obs: np.ndarray,
                     margins_obs: np.ndarray, p2f: np.ndarray,
                     cross_ratio: float = 0.1) -> dict:
    """Calibrate linear demand q = a - B p."""
    J = len(shares_obs)
    mc = prices_obs * (1.0 - margins_obs)
    omega = ownership_matrix(p2f)
    M = 1.0
    q_obs = shares_obs * M
    markups = prices_obs - mc
    b_own = q_obs / markups
    B = np.zeros((J, J))
    for j in range(J):
        B[j, j] = b_own[j]
        for k in range(J):
            if k != j:
                B[j, k] = -cross_ratio * np.sqrt(b_own[j] * b_own[k])
    for j in range(J):
        cross_contrib = 0.0
        for k in range(J):
            if k != j and omega[j, k] == 1:
                cross_contrib += (-B[k, j]) * markups[k]
        B[j, j] = (q_obs[j] + cross_contrib) / markups[j]
    a = q_obs + B @ prices_obs
    return {"a": a, "B": B, "mc": mc}


def calibrate_loglinear(shares_obs: np.ndarray, prices_obs: np.ndarray,
                        margins_obs: np.ndarray, p2f: np.ndarray,
                        cross_elas: float = 0.3) -> dict:
    """Calibrate log-linear demand log q_j = a_j + sum E_{jk} log p_k."""
    J = len(shares_obs)
    mc = prices_obs * (1.0 - margins_obs)
    omega = ownership_matrix(p2f)
    M = 1.0
    q_obs = shares_obs * M
    markups = prices_obs - mc
    e_own = -prices_obs / markups
    E = np.zeros((J, J))
    for j in range(J):
        E[j, j] = e_own[j]
        for k in range(J):
            if k != j:
                E[j, k] = cross_elas
    for j in range(J):
        cross_contrib = 0.0
        for k in range(J):
            if k != j and omega[j, k] == 1:
                cross_contrib += q_obs[k] * E[k, j] / prices_obs[j] * markups[k]
        E[j, j] = -(q_obs[j] + cross_contrib) * prices_obs[j] / (q_obs[j] * markups[j])
    a_ll = np.log(q_obs) - E @ np.log(prices_obs)
    return {"a_ll": a_ll, "E": E, "mc": mc}


# =========================================================================
# Screening metrics (UPP, GUPPI, CMCR) and welfare
# =========================================================================

def diversion_ratios_from_jacobian(dqdp: np.ndarray) -> np.ndarray:
    """Diversion ratio D_{j->k} = -(dq_k / dp_j) / (dq_j / dp_j)."""
    J = dqdp.shape[0]
    D = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j != k:
                D[j, k] = -dqdp[k, j] / dqdp[j, j]
    return D


def diversion_ratios_logit(shares: np.ndarray) -> np.ndarray:
    """Closed-form logit diversion D_{j->k} = s_k / (1 - s_j) for j != k."""
    J = len(shares)
    D = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j != k:
                D[j, k] = shares[k] / (1.0 - shares[j])
    return D


def compute_upp(D: np.ndarray, prices: np.ndarray, mc: np.ndarray,
                p2f_pre: np.ndarray, p2f_post: np.ndarray) -> np.ndarray:
    """Upward Pricing Pressure: UPP_j = sum over newly co-owned k of D_{j->k} (p_k - c_k)."""
    J = len(prices)
    upp = np.zeros(J)
    for j in range(J):
        for k in range(J):
            if j != k and p2f_post[j] == p2f_post[k] and p2f_pre[j] != p2f_pre[k]:
                upp[j] += D[j, k] * (prices[k] - mc[k])
    return upp


def compute_guppi(D, prices, mc, p2f_pre, p2f_post):
    return compute_upp(D, prices, mc, p2f_pre, p2f_post) / prices


def compute_cmcr(D, prices, mc, p2f_pre, p2f_post):
    return compute_upp(D, prices, mc, p2f_pre, p2f_post) / mc


def producer_surplus(p: np.ndarray, q: np.ndarray, mc: np.ndarray) -> float:
    return float(np.sum((p - mc) * q))


def fmt_vector(values: np.ndarray, digits: int = 2) -> str:
    return "[" + ", ".join(f"{float(v):.{digits}f}" for v in values) + "]"


def first_zero_crossing(x: np.ndarray, y: list) -> float:
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
    setup_style()

    # =====================================================================
    # PART 1. Concentration screens (HHI, delta-HHI, effective firms)
    # =====================================================================
    print("=" * 72)
    print("PART 1. Concentration screens")
    print("=" * 72)

    markets = {
        "Perfect competition (100 firms)": equal_shares(100),
        "10 equal firms": equal_shares(10),
        "5 equal firms": equal_shares(5),
        "Asymmetric (40-30-20-10)": np.array([0.40, 0.30, 0.20, 0.10]),
        "Duopoly (50-50)": np.array([0.50, 0.50]),
        "Dominant firm (70-10-10-10)": np.array([0.70, 0.10, 0.10, 0.10]),
        "Near-monopoly (90-5-5)": np.array([0.90, 0.05, 0.05]),
        "Monopoly": np.array([1.0]),
    }
    market_rows = []
    for name, shares in markets.items():
        hhi = compute_hhi(shares)
        market_rows.append({
            "Market Structure": name,
            "N Firms": len(shares),
            "Top Share (%)": f"{shares.max() * 100:.0f}",
            "HHI": f"{hhi:.0f}",
            "Effective N": f"{effective_firms(shares):.2f}",
            "Classification": classify_hhi(hhi),
        })
    df_markets = pd.DataFrame(market_rows)

    n_firms_range = np.arange(1, 51)
    hhi_equal = np.array([compute_hhi(equal_shares(n)) for n in n_firms_range])

    merger_cases = {
        "10 equal firms\n(merge 2 of 10)": equal_shares(10),
        "5 equal firms\n(merge 2 of 5)": equal_shares(5),
        "Asymmetric\n40-30-20-10": np.array([0.40, 0.30, 0.20, 0.10]),
        "Duopoly\n50-50": np.array([0.50, 0.50]),
        "Dominant\n70-10-10-10": np.array([0.70, 0.10, 0.10, 0.10]),
    }
    merger_results = []
    for label, shares in merger_cases.items():
        sorted_s = np.sort(shares)[::-1]
        s1, s2 = sorted_s[0], sorted_s[1]
        hhi_before = compute_hhi(shares)
        d_hhi = delta_hhi(s1, s2)
        merger_results.append({
            "label": label,
            "hhi_before": hhi_before,
            "delta_hhi": d_hhi,
            "hhi_after": hhi_before + d_hhi,
        })

    print(df_markets.to_string(index=False))
    print()

    # =====================================================================
    # PART 2. Four-product Bertrand-Nash baseline (logit, single margin)
    # =====================================================================
    print("=" * 72)
    print("PART 2. Four-product baseline (logit, one observed margin)")
    print("=" * 72)

    s4 = np.array([0.15, 0.15, 0.30, 0.30])
    p4 = np.array([1.0, 1.0, 1.0, 1.0])
    p2f4_pre = np.array([1, 2, 3, 4])
    p2f4_post = np.array([1, 1, 3, 4])
    margin4 = 0.50
    n4 = len(s4)
    omega4_pre = ownership_matrix(p2f4_pre)
    omega4_post = ownership_matrix(p2f4_post)

    cal4 = calibrate_logit_simple(s4, p4, margin4, p2f4_pre)
    alpha4, xi4, mc4 = cal4["alpha"], cal4["xi"], cal4["mc"]
    foc4_check = foc_logit(p4, mc4, alpha4, xi4, omega4_pre)

    p4_post = scipy.optimize.fsolve(
        foc_logit, x0=p4 * 1.1, args=(mc4, alpha4, xi4, omega4_post),
    )
    s4_post = shares_logit(p4_post, alpha4, xi4)

    mc4_eff = mc4 * np.array([0.9, 0.9, 1.0, 1.0])
    p4_post_eff = scipy.optimize.fsolve(
        foc_logit, x0=p4 * 1.1, args=(mc4_eff, alpha4, xi4, omega4_post),
    )
    s4_post_eff = shares_logit(p4_post_eff, alpha4, xi4)

    div4 = diversion_ratios_logit(s4)

    scenarios4 = {
        "Pre-merger": {"prices": p4, "shares": s4, "owners": p2f4_pre, "mc": mc4},
        "Merger 1+2": {"prices": p4_post, "shares": s4_post, "owners": p2f4_post, "mc": mc4},
        "Merger 1+2, 10% cost reduction": {
            "prices": p4_post_eff, "shares": s4_post_eff, "owners": p2f4_post, "mc": mc4_eff,
        },
    }

    print(f"alpha (logit)      = {alpha4:.4f}")
    print(f"marginal costs     = {mc4}")
    print(f"FOC residual       = {np.max(np.abs(foc4_check)):.2e}")
    print(f"Post-merger prices = {p4_post}")
    print(f"With 10% mc cut    = {p4_post_eff}")
    print()

    # =====================================================================
    # PART 3. Six-product extension (three demand systems)
    # =====================================================================
    print("=" * 72)
    print("PART 3. Six-product extension across logit, linear, log-linear")
    print("=" * 72)

    J = 6
    shares_obs = np.array([0.12, 0.10, 0.15, 0.13, 0.08, 0.07])
    prices_obs = np.array([1.0, 1.2, 0.9, 1.1, 1.3, 1.4])
    margins_obs = np.array([0.40, 0.35, 0.45, 0.40, 0.30, 0.28])
    p2f_pre = np.array([1, 1, 2, 2, 3, 3])
    p2f_post = np.array([1, 1, 1, 1, 3, 3])
    M = 1.0
    product_names = [f"P{j+1}" for j in range(J)]

    omega_pre = ownership_matrix(p2f_pre)
    omega_post = ownership_matrix(p2f_post)

    cal_logit = calibrate_logit(shares_obs, prices_obs, margins_obs, p2f_pre)
    cal_linear = calibrate_linear(shares_obs, prices_obs, margins_obs, p2f_pre, cross_ratio=0.1)
    cal_loglinear = calibrate_loglinear(shares_obs, prices_obs, margins_obs, p2f_pre, cross_elas=0.15)

    foc_check_logit = foc_logit(prices_obs, cal_logit["mc"], cal_logit["alpha"],
                                cal_logit["xi"], omega_pre)
    foc_check_linear = foc_linear(prices_obs, cal_linear["mc"], cal_linear["a"],
                                  cal_linear["B"], omega_pre)
    foc_check_loglinear = foc_loglinear(prices_obs, cal_loglinear["mc"], cal_loglinear["a_ll"],
                                        cal_loglinear["E"], omega_pre)

    jac_logit = jacobian_logit(prices_obs, cal_logit["alpha"], cal_logit["xi"])
    jac_linear = jacobian_linear(prices_obs, cal_linear["a"], cal_linear["B"])
    jac_loglinear = jacobian_loglinear(prices_obs, cal_loglinear["a_ll"], cal_loglinear["E"])

    div_logit = diversion_ratios_from_jacobian(jac_logit)
    div_linear = diversion_ratios_from_jacobian(jac_linear)
    div_loglinear = diversion_ratios_from_jacobian(jac_loglinear)

    mc_logit, mc_linear, mc_loglinear = cal_logit["mc"], cal_linear["mc"], cal_loglinear["mc"]
    upp_logit = compute_upp(div_logit, prices_obs, mc_logit, p2f_pre, p2f_post)
    upp_linear = compute_upp(div_linear, prices_obs, mc_linear, p2f_pre, p2f_post)
    upp_loglinear = compute_upp(div_loglinear, prices_obs, mc_loglinear, p2f_pre, p2f_post)
    guppi_logit = compute_guppi(div_logit, prices_obs, mc_logit, p2f_pre, p2f_post)
    guppi_linear = compute_guppi(div_linear, prices_obs, mc_linear, p2f_pre, p2f_post)
    guppi_loglinear = compute_guppi(div_loglinear, prices_obs, mc_loglinear, p2f_pre, p2f_post)
    cmcr_logit = compute_cmcr(div_logit, prices_obs, mc_logit, p2f_pre, p2f_post)
    cmcr_linear = compute_cmcr(div_linear, prices_obs, mc_linear, p2f_pre, p2f_post)
    cmcr_loglinear = compute_cmcr(div_loglinear, prices_obs, mc_loglinear, p2f_pre, p2f_post)

    p_post_logit = scipy.optimize.fsolve(
        foc_logit, x0=prices_obs * 1.05,
        args=(cal_logit["mc"], cal_logit["alpha"], cal_logit["xi"], omega_post),
    )
    s_post_logit = shares_logit(p_post_logit, cal_logit["alpha"], cal_logit["xi"])

    p_post_linear = scipy.optimize.fsolve(
        foc_linear, x0=prices_obs * 1.05,
        args=(cal_linear["mc"], cal_linear["a"], cal_linear["B"], omega_post),
    )
    q_post_linear = shares_linear(p_post_linear, cal_linear["a"], cal_linear["B"])

    logp_post_loglinear = scipy.optimize.fsolve(
        foc_loglinear_logp, x0=np.log(prices_obs * 1.05),
        args=(cal_loglinear["mc"], cal_loglinear["a_ll"], cal_loglinear["E"], omega_post),
    )
    p_post_loglinear = np.exp(logp_post_loglinear)
    q_post_loglinear = shares_loglinear(p_post_loglinear, cal_loglinear["a_ll"], cal_loglinear["E"])

    foc_post_logit = foc_logit(p_post_logit, cal_logit["mc"], cal_logit["alpha"],
                               cal_logit["xi"], omega_post)
    foc_post_linear = foc_linear(p_post_linear, cal_linear["mc"], cal_linear["a"],
                                 cal_linear["B"], omega_post)
    foc_post_loglinear = foc_loglinear(p_post_loglinear, cal_loglinear["mc"], cal_loglinear["a_ll"],
                                       cal_loglinear["E"], omega_post)
    post_residuals = {
        "Logit": np.max(np.abs(foc_post_logit)),
        "Linear": np.max(np.abs(foc_post_linear)),
        "Log-linear": np.max(np.abs(foc_post_loglinear)),
    }

    dp_logit = (p_post_logit - prices_obs) / prices_obs * 100
    dp_linear = (p_post_linear - prices_obs) / prices_obs * 100
    dp_loglinear = (p_post_loglinear - prices_obs) / prices_obs * 100

    cs_pre_logit = cs_logit(prices_obs, cal_logit["alpha"], cal_logit["xi"], M)
    ps_pre_logit = producer_surplus(prices_obs, shares_obs * M, mc_logit)
    cs_post_logit_v = cs_logit(p_post_logit, cal_logit["alpha"], cal_logit["xi"], M)
    ps_post_logit = producer_surplus(p_post_logit, s_post_logit * M, mc_logit)

    q_pre_linear = shares_linear(prices_obs, cal_linear["a"], cal_linear["B"])
    cs_pre_linear = cs_linear(prices_obs, cal_linear["a"], cal_linear["B"],
                              np.linalg.solve(cal_linear["B"], cal_linear["a"]))
    ps_pre_linear = producer_surplus(prices_obs, q_pre_linear, mc_linear)
    cs_post_linear_v = cs_linear(p_post_linear, cal_linear["a"], cal_linear["B"],
                                 np.linalg.solve(cal_linear["B"], cal_linear["a"]))
    ps_post_linear = producer_surplus(p_post_linear, q_post_linear, mc_linear)

    q_pre_loglinear = shares_loglinear(prices_obs, cal_loglinear["a_ll"], cal_loglinear["E"])
    p_high_ll = prices_obs * 5.0
    cs_pre_loglinear = cs_loglinear(prices_obs, p_high_ll, cal_loglinear["a_ll"], cal_loglinear["E"])
    ps_pre_loglinear = producer_surplus(prices_obs, q_pre_loglinear, mc_loglinear)
    cs_post_loglinear_v = cs_loglinear(p_post_loglinear, p_high_ll, cal_loglinear["a_ll"], cal_loglinear["E"])
    ps_post_loglinear = producer_surplus(p_post_loglinear, q_post_loglinear, mc_loglinear)

    dCS = {"Logit": cs_post_logit_v - cs_pre_logit,
           "Linear": cs_post_linear_v - cs_pre_linear,
           "Log-linear": cs_post_loglinear_v - cs_pre_loglinear}
    dPS = {"Logit": ps_post_logit - ps_pre_logit,
           "Linear": ps_post_linear - ps_pre_linear,
           "Log-linear": ps_post_loglinear - ps_pre_loglinear}
    dW = {k: dCS[k] + dPS[k] for k in dCS}

    print(f"FOC residuals: logit {np.max(np.abs(foc_check_logit)):.1e}, "
          f"linear {np.max(np.abs(foc_check_linear)):.1e}, "
          f"log-linear {np.max(np.abs(foc_check_loglinear)):.1e}")
    print(f"Logit post-merger prices:      {p_post_logit}")
    print(f"Linear post-merger prices:     {p_post_linear}")
    print(f"Log-linear post-merger prices: {p_post_loglinear}")
    for m in ["Logit", "Linear", "Log-linear"]:
        print(f"  {m:12s}: dCS={dCS[m]:+.4f}  dPS={dPS[m]:+.4f}  dW={dW[m]:+.4f}")
    print()

    # =====================================================================
    # Build the report
    # =====================================================================

    report = ModelReport(
        "Merger Pricing: Concentration Screens, Diversion, and Bertrand-Nash Equilibrium",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A horizontal merger between close substitutes raises three questions in order. "
        "How concentrated is the market already? What happens to prices once two products "
        "share an owner? Does the price answer survive a different assumption about how "
        "customers substitute?\n\n"
        "The first question is fast. HHI and delta-HHI come straight off the sales table. "
        "They use ownership shares and nothing else.\n\n"
        "The second question needs structure. We calibrate logit demand from one observed "
        "margin. Then we change the ownership matrix and resolve the Bertrand-Nash pricing "
        "FOC for new prices.\n\n"
        "The third question is harder. Different demand systems can match the same observed "
        "market and still disagree about counterfactual prices. We calibrate three of them. "
        "We compute UPP, GUPPI, and CMCR screens at observed prices, solve full post-merger "
        "equilibria, and ask how much marginal-cost reduction would offset the price effect.\n\n"
        "The same logic threads the three layers. Two competitors come under one owner. "
        "The owner now internalizes diversion between them. Whether prices rise, and by how "
        "much, depends on substitution to rivals, the outside option, and any cost savings "
        "the merger brings."
    )

    report.add_equations(r"""
The three layers stack on the same observed market.

### A. Concentration screens

Let firms be indexed by $f=1,\ldots,F$ with market shares $s_f$ summing to one.
The Herfindahl-Hirschman Index is

$$\text{HHI}=10{,}000\sum_{f=1}^{F}s_f^2.$$

The associated effective number of equal-sized firms is

$$N_{\text{eff}}=\frac{1}{\sum_f s_f^2}=\frac{10{,}000}{\text{HHI}}.$$

If firms $a$ and $b$ merge while quantities are held fixed,

$$\Delta\text{HHI}=10{,}000\left[(s_a+s_b)^2-s_a^2-s_b^2\right]=20{,}000\,s_a s_b.$$

Worked example. Take three firms with shares $(0.5, 0.3, 0.2)$. The HHI is
$10{,}000(0.25+0.09+0.04)=3{,}800$. The effective firm count is
$10{,}000/3{,}800\approx 2.63$, well under the three nominal firms. A merger of
firms 1 and 2 raises HHI by $20{,}000\cdot 0.5\cdot 0.3=3{,}000$ to $6{,}800$.
That moves the market well into the highly-concentrated band.

The 2023 DOJ/FTC Merger Guidelines treat HHI above 1,800 as highly concentrated.
A delta-HHI above 100 is significant for the structural presumption.

### B. Bertrand-Nash pricing with multi-product firms

There are $J$ inside products. Product $j$ has price $p_j$, marginal cost $c_j$,
quantity or share $q_j(p)$, and owner $f(j)$. The ownership matrix records
which products belong to the same firm:

$$\Omega_{jk}=\mathbf 1\{f(j)=f(k)\}.$$

Worked example. Take $J=4$ products with pre-merger owners $f^{\text{pre}}=(1,2,3,4)$,
so each product is its own single-product firm:

$$\Omega^{\text{pre}}=\begin{pmatrix}1&0&0&0\\0&1&0&0\\0&0&1&0\\0&0&0&1\end{pmatrix}.$$

Now firm 1 buys firm 2. The post-merger owners are $f^{\text{post}}=(1,1,3,4)$.
Products 1 and 2 share an owner. The off-diagonal entries that link them switch
on:

$$\Omega^{\text{post}}=\begin{pmatrix}1&1&0&0\\1&1&0&0\\0&0&1&0\\0&0&0&1\end{pmatrix}.$$

Those two new ones in the upper-left block are the entire merger as the model
sees it. Every price effect that follows is the consequence of switching them
on.

A multi-product Bertrand firm chooses each price to satisfy

$$0=q_j(p)+\sum_{k=1}^J \Omega_{jk}(p_k-c_k)\frac{\partial q_k(p)}{\partial p_j}, \qquad j=1,\ldots,J.$$

With $\Delta_{kj}(p)=\partial q_k(p)/\partial p_j$, the vector form is

$$q(p)+\underbrace{(\Omega\circ \Delta(p)^\top)}_{\text{within-firm price externality}}(p-c)=0.$$

Here $\circ$ is the element-wise (Hadamard) product. Multiplying by $\Omega$
knocks out cross-firm price externalities. Only co-owned products show up in
any one firm's markup equation. The merger flips entries of $\Omega$ from zero
to one. The FOCs then solve for new prices.

### C. Three demand systems

The same observed market can give very different counterfactual prices once we
change the demand curvature. We calibrate three demand systems. Each one matches
the observed shares and prices by construction. Linear and log-linear demand
carry a full $J\times J$ slope matrix, so they also reproduce the entire
observed margin vector exactly. Logit demand has a single price coefficient
$\alpha$. We pin it down from the average single-product margin condition and
recover marginal costs from the pre-merger FOC, so the logit FOC residual is
zero at calibration while its implied margins track the observed margins only
as closely as one coefficient allows.

Logit shares are

$$s_j^{L}(p)=\frac{\exp(\xi_j+\alpha p_j)}{1+\sum_{\ell=1}^J \exp(\xi_\ell+\alpha p_\ell)}, \qquad \alpha<0,$$

where $\xi_j$ is the mean indirect utility of product $j$ from non-price
characteristics. The closed-form Jacobian is

$$\Delta^{L}_{jk}=\alpha\,s_j(\mathbf 1\{j=k\}-s_k).$$

Opened element by element for $J=4$ this reads

$$\Delta^{L}=\alpha\begin{pmatrix}s_1(1-s_1)&-s_1 s_2&-s_1 s_3&-s_1 s_4\\-s_1 s_2&s_2(1-s_2)&-s_2 s_3&-s_2 s_4\\-s_1 s_3&-s_2 s_3&s_3(1-s_3)&-s_3 s_4\\-s_1 s_4&-s_2 s_4&-s_3 s_4&s_4(1-s_4)\end{pmatrix}.$$

The diagonal carries the own-price effect. Off-diagonals carry cross-price
effects. The whole matrix scales with $\alpha$.

Linear demand is

$$q_j^{A}(p)=a_j-\sum_{k=1}^J B_{jk}p_k, \qquad \Delta^{A}=-B,$$

so $\Delta^{A}$ does not depend on $p$. Here $a_j$ is the demand intercept and
$B_{jk}$ is the price-response matrix.

Log-linear demand is

$$\log q_j^{E}(p)=a_j^E+\sum_{k=1}^J E_{jk}\log p_k, \qquad \Delta^{E}_{jk}=q_j\,E_{jk}/p_k,$$

so the elasticities $E_{jk}$ are constant and the slopes scale with quantities
and inverse prices.

### D. Diversion ratios and screening metrics

The local diversion ratio from product $j$ to product $k$ is

$$D_{j\to k}=-\frac{\partial q_k(p)/\partial p_j}{\partial q_j(p)/\partial p_j}, \qquad j\neq k.$$

Under simple logit it collapses to $D_{j\to k}=s_k/(1-s_j)$ and depends only on
shares and the outside option.

For products that become newly co-owned after the merger,

$$\text{UPP}_j=\sum_{k:\Omega^{\text{post}}_{jk}=1,\ \Omega^{\text{pre}}_{jk}=0} D_{j\to k}(p_k-c_k),$$

with

$$\text{GUPPI}_j=\frac{\text{UPP}_j}{p_j}, \qquad \text{CMCR}_j=\frac{\text{UPP}_j}{c_j}.$$

GUPPI is a first-order screen evaluated at observed prices. It says how much
upward pricing pressure the merger generates locally. CMCR reports the
marginal-cost reduction that would offset that pressure at those same prices.
Neither metric clears the market. The simulation in Method 3 does. It solves
the full pricing system under post-merger ownership.
""")

    report.add_model_setup(
        "Two market setups run in sequence. The first is a small four-product market. We "
        "calibrate logit demand from one observed margin. The second is a six-product "
        "market. We calibrate three demand systems against the full margin vector.\n\n"
        "**Four-product baseline (Part B).**\n\n"
        "| Object | Value | Role |\n"
        "|--------|-------|------|\n"
        f"| Inside products | {n4} | Four single-product firms before the merger |\n"
        f"| Inside shares | {fmt_vector(s4)} | Observed product shares |\n"
        f"| Outside share | {1.0 - np.sum(s4):.2f} | No-purchase option |\n"
        f"| Prices | {fmt_vector(p4)} | Pre-merger prices |\n"
        f"| Margin (product 1) | {margin4:.2f} | Pins down the logit price coefficient |\n"
        f"| $\\alpha$ | {alpha4:.4f} | Calibrated price sensitivity |\n"
        f"| Marginal costs | {fmt_vector(mc4)} | Recovered from the pre-merger FOCs |\n"
        "| Scenarios | merger 1+2; merger 1+2 with 10% cost reduction | Ownership and cost experiments |\n\n"
        "**Six-product extended (Part C).**\n\n"
        "| Parameter | Value | Description |\n"
        "|-----------|-------|-------------|\n"
        f"| Products $J$ | {J} | 3 firms, 2 products each |\n"
        f"| Shares | {fmt_vector(shares_obs)} | Pre-merger inside shares |\n"
        f"| Prices | {fmt_vector(prices_obs)} | Pre-merger prices |\n"
        f"| Margins | {fmt_vector(margins_obs)} | Price-cost margins by product |\n"
        f"| Outside share | {1 - np.sum(shares_obs):.2f} | Outside option in the logit demand system |\n"
        f"| $\\alpha$ (logit) | {cal_logit['alpha']:.4f} | Calibrated price coefficient |\n"
        "| Linear cross-slope ratio | 0.10 | Cross-slope relative to geometric mean own-slope |\n"
        "| Log-linear cross elasticity | 0.15 | Maintained symmetric cross-price elasticity |\n"
        "| Merger | Firm 1 buys Firm 2 | Products 1-4 move under common ownership |\n"
        "| Benchmark | Post-merger Bertrand-Nash FOC | Equilibrium used to judge first-order screens |"
    )

    report.add_solution_method(
        "Three methods run in sequence. Each one answers a question the previous method "
        "cannot reach. The defining formulas all live in the Equations section. Here we "
        "reuse the same symbols inside the pseudocode.\n\n"
        "### Method 1: HHI concentration screens\n\n"
        "HHI squares the ownership shares. The largest firm dominates the index. Two small "
        "firms barely move it. So the merger increment cares only about the two merging "
        "shares, not about anyone else. There is nothing to iterate. The cost is linear in "
        "the firm count and the answer is exact.\n\n"
        "```text\n"
        "Inputs:  firm shares s = (s_1, ..., s_F), candidate merger pair (a, b)\n"
        "Outputs: HHI, N_eff, delta-HHI, structural classification\n"
        "\n"
        "1. HHI       = 10000 * sum_{f=1..F} s_f^2\n"
        "2. N_eff     = 10000 / HHI\n"
        "3. delta-HHI = 20000 * s_a * s_b\n"
        "4. classify HHI:  < 1000        -> Unconcentrated\n"
        "                  1000 to 1800  -> Moderate\n"
        "                  > 1800        -> Highly Concentrated\n"
        "```\n\n"
        "The screen says nothing about prices. Two products that share an owner but do not "
        "substitute can move HHI without moving prices at all. Method 2 closes that gap.\n\n"
        "### Method 2: Four-product Bertrand-Nash baseline (logit, single margin)\n\n"
        "One product's margin is the whole calibration. The single-product FOC for product 1 "
        "pins down the price coefficient. The other marginal costs come from inverting the "
        "multi-product FOC at observed prices. Post-merger prices come from a Newton-type "
        "root finder on the FOC under the new ownership matrix. Convergence is locally "
        "quadratic when the warm-start sits close to the solution.\n\n"
        "```text\n"
        "Inputs:  shares s, prices p, margin m_1 on product 1, owners f^pre, f^post\n"
        "Outputs: alpha, xi, mc, post-merger prices p^post and shares s^post\n"
        "\n"
        "1. c_1   = p_1 * (1 - m_1)\n"
        "2. alpha = -1 / [(1 - s_1) * (p_1 - c_1)]                  # single-product FOC\n"
        "3. xi_j  = log(s_j / s_0) - alpha * p_j,    s_0 = 1 - sum_j s_j\n"
        "4. mc    = p + (Omega^pre .* Delta(p)^T)^{-1} s            # multi-product FOC inversion\n"
        "5. build Omega^post from f^post\n"
        "6. solve  s(p) + (Omega^post .* Delta(p)^T) (p - mc) = 0   # fsolve, warm-start p^0 = 1.1 * p\n"
        "7. (optional)  mc' = mc with merging-product entries scaled by (1 - eps); resolve\n"
        "```\n\n"
        "The root finder can fail when the warm-start is too far from the equilibrium or "
        "when the Jacobian is ill-conditioned. The implementation reports the FOC residual "
        "after each solve. A residual near machine precision confirms that the reported "
        "prices clear the system.\n\n"
        "### Method 3: Six-product extension with three demand systems\n\n"
        "Now we run the same pricing FOC across three demand systems. Each one matches the "
        "same observed shares and prices by construction; linear and log-linear demand also "
        "match the full observed margin vector, while logit matches margins only as closely "
        "as one price coefficient allows. Any disagreement on counterfactual "
        "prices is therefore curvature, not data. UPP, GUPPI, and CMCR are local screens "
        "evaluated at observed prices. The post-merger FOC adds rival reactions and "
        "pass-through. The efficiency frontier sweeps a cost-reduction grid and re-solves "
        "at each step, warm-starting each solve from the previous one. That keeps the path "
        "monotone and the iteration count low.\n\n"
        "```text\n"
        "Inputs:  shares s, prices p, margins m, owners f^pre, f^post\n"
        "Outputs: per-system screens, post-merger prices, welfare, efficiency frontier\n"
        "\n"
        "1. for each demand system d in {L (logit), A (linear), E (log-linear)}:\n"
        "     a. fit theta_d so that q_d(p; theta_d) = s\n"
        "     b. mc_d   = p + (Omega^pre .* Delta_d(p)^T)^{-1} q_d(p)         # FOC inversion\n"
        "     c. D_d[j, k] = -Delta_d[k, j] / Delta_d[j, j],   j != k         # diversion\n"
        "     d. for each j newly co-owned:\n"
        "          UPP_j   = sum_{k newly co-owned} D_d[j, k] * (p_k - mc_d,k)\n"
        "          GUPPI_j = UPP_j / p_j,    CMCR_j = UPP_j / mc_d,j\n"
        "     e. solve  q_d(p^post) + (Omega^post .* Delta_d(p^post)^T)\n"
        "                              * (p^post - mc_d) = 0                 # fsolve\n"
        "     f. dCS_d = CS_d(p^post) - CS_d(p),    dPS_d = PS_d(p^post) - PS_d(p)\n"
        "\n"
        "2. for eps in {0, 0.005, ..., 0.6}:\n"
        "     mc_eff      = mc_d with merging-product entries scaled by (1 - eps)\n"
        "     p^post(eps) = solve post-merger FOC with mc_d replaced by mc_eff\n"
        "                   (warm-start from previous eps)\n"
        "     record  avg_dp(eps) = mean over j in {1..4} of (p^post_j(eps) - p_j) / p_j\n"
        "\n"
        "3. eps* = first zero of avg_dp(eps)         # break-even cost reduction\n"
        "```\n\n"
        "Log-linear demand can drive prices negative during root-finder iterations. We solve "
        "in log-price space to keep prices positive at every step. Linear demand can return "
        "negative quantities at extreme prices, so we clip at a small positive value. Both "
        "safeguards leave the equilibrium unchanged where the unmodified solver also "
        "converges.\n\n"
        f"FOC residuals after calibration: four-product baseline "
        f"{np.max(np.abs(foc4_check)):.1e}; six-product extended "
        f"(logit {np.max(np.abs(foc_check_logit)):.1e}, "
        f"linear {np.max(np.abs(foc_check_linear)):.1e}, "
        f"log-linear {np.max(np.abs(foc_check_loglinear)):.1e}). "
        "All calibrations reproduce the observed pricing conditions to numerical precision."
    )

    # =====================================================================
    # FIGURES AND RESULTS
    # =====================================================================

    # ---------- Part 1 results: concentration screens ----------
    report.add_results(
        "### Part 1: Concentration screens\n\n"
        "A sales table tells us how concentrated ownership already is. It also tells us "
        "how much a candidate merger would shift the index. The HHI-vs-firms curve below "
        "puts symmetric and asymmetric markets on the same scale. The delta-HHI bars show "
        "how merger arithmetic depends on the absolute size of the merging shares."
    )

    fig_hhi, ax_hhi = plt.subplots(figsize=(9, 5))
    ax_hhi.plot(n_firms_range, hhi_equal, "b-", linewidth=2)
    ax_hhi.axhspan(0, 1000, alpha=0.10, color="green", label="Unconcentrated (< 1000)")
    ax_hhi.axhspan(1000, 1800, alpha=0.10, color="orange", label="Moderate (1000-1800)")
    ax_hhi.axhspan(1800, 10500, alpha=0.10, color="red", label="Highly Concentrated (> 1800)")
    ax_hhi.set_xlabel("Number of Equal-Sized Firms ($N$)")
    ax_hhi.set_ylabel("HHI")
    ax_hhi.set_title("HHI on the Equal-Sized-Firm Scale")
    ax_hhi.set_xlim(1, 50)
    ax_hhi.set_ylim(0, 10500)
    ax_hhi.legend(loc="upper right", fontsize=9)
    for n_mark in [2, 4, 7, 10]:
        hhi_mark = compute_hhi(equal_shares(n_mark))
        ax_hhi.annotate(
            f"N={n_mark}\nHHI={hhi_mark:.0f}",
            xy=(n_mark, hhi_mark),
            xytext=(n_mark + 3, hhi_mark + 500),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )
    report.add_figure(
        "figures/hhi-vs-nfirms.png",
        "HHI as a function of the number of equal-sized firms",
        fig_hhi,
        description="For symmetric firms, HHI is exactly $10{,}000/N$. Most of the index "
        "movement happens between monopoly and five equal firms. The highly-concentrated "
        "threshold of 1,800 lines up with about 5.6 equal-sized firms.",
    )

    fig_dhhi, ax_dhhi = plt.subplots(figsize=(10, 5))
    x_dh = np.arange(len(merger_results))
    width_dh = 0.35
    ax_dhhi.bar(
        x_dh - width_dh / 2,
        [m["hhi_before"] for m in merger_results],
        width_dh,
        label="HHI before",
        color="#4878CF",
        edgecolor="white",
    )
    ax_dhhi.bar(
        x_dh + width_dh / 2,
        [m["hhi_after"] for m in merger_results],
        width_dh,
        label="HHI after",
        color="#D65F5F",
        edgecolor="white",
    )
    for i, m in enumerate(merger_results):
        ax_dhhi.text(
            i + width_dh / 2, m["hhi_after"] + 80,
            f"$\\Delta$={m['delta_hhi']:.0f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax_dhhi.axhline(y=1000, color="green", linestyle="--", linewidth=1, alpha=0.7)
    ax_dhhi.axhline(y=1800, color="orange", linestyle="--", linewidth=1, alpha=0.7)
    ax_dhhi.text(len(merger_results) - 0.5, 1060, "Unconcentrated (1000)", fontsize=8, color="green")
    ax_dhhi.text(len(merger_results) - 0.5, 1860, "Highly concentrated (1800)", fontsize=8, color="orange")
    ax_dhhi.set_xticks(x_dh)
    ax_dhhi.set_xticklabels([m["label"] for m in merger_results], fontsize=9)
    ax_dhhi.set_ylabel("HHI")
    ax_dhhi.set_title("HHI Before and After Merger of the Two Largest Firms")
    ax_dhhi.legend()
    fig_dhhi.tight_layout()
    report.add_figure(
        "figures/delta-hhi.png",
        "HHI before and after merging the two largest firms",
        fig_dhhi,
        description="The merger bars are pure index arithmetic. The same formula, "
        "$20{,}000\\,s_a s_b$, makes a 40-30 merger far larger than a merger of two small "
        "firms. That scale is why agencies reach for HHI before estimating demand.",
    )

    report.add_table(
        "tables/market-hhi.csv",
        "HHI for example market structures",
        df_markets,
        description="The effective firm count exposes asymmetry. A 70-10-10-10 market has "
        "four firms. Its HHI of 5,200 is equivalent to fewer than two equal-sized firms.",
    )

    # ---------- Part 2 results: four-product baseline ----------
    report.add_results(
        "### Part 2: Four-product baseline\n\n"
        "Concentration says nothing about prices. We need a calibrated demand system to "
        "talk about counterfactual prices. The simplest useful case is logit demand on a "
        "four-product market with one observed margin. The merger puts products 1 and 2 "
        "under one owner. Products 3 and 4 stay outside the deal."
    )

    fig_p4, ax_p4 = plt.subplots(figsize=(9, 5))
    x4 = np.arange(n4)
    scenario_names = list(scenarios4.keys())
    offsets4 = (np.arange(len(scenario_names)) - (len(scenario_names) - 1) / 2) * 0.22
    width4 = 0.22
    colors_s4 = ["steelblue", "coral", "seagreen"]
    for i, name in enumerate(scenario_names):
        p_s = scenarios4[name]["prices"]
        ax_p4.bar(x4 + offsets4[i], p_s, width4, label=name, color=colors_s4[i])
    ax_p4.axhline(np.mean(p4), color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax_p4.set_xlabel("Product")
    ax_p4.set_ylabel("Price")
    ax_p4.set_title("Four-Product Baseline: Equilibrium Prices Under Alternative Ownership")
    ax_p4.set_xticks(x4)
    ax_p4.set_xticklabels([f"P{j+1}" for j in range(n4)])
    ymax = max(np.max(v["prices"]) for v in scenarios4.values()) * 1.12
    ax_p4.set_ylim(0, ymax)
    ax_p4.legend(frameon=False, ncol=2, fontsize=9)
    report.add_figure(
        "figures/four-product-prices.png",
        "Equilibrium prices in the four-product baseline under alternative ownership",
        fig_p4,
        description="Once products 1 and 2 share an owner, that owner raises both prices. "
        "A lost sale to the partner product is no longer fully lost. The 10% cost cut "
        "mutes the price increase but does not erase it. Products 3 and 4 also rise as "
        "rivals best-respond.",
    )

    fig_d4, ax_d4 = plt.subplots(figsize=(6, 5))
    div4_plot = div4.copy()
    np.fill_diagonal(div4_plot, np.nan)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#f0f0f0")
    im = ax_d4.imshow(div4_plot, cmap=cmap, vmin=0, vmax=np.nanmax(div4_plot))
    ax_d4.set_xticks(range(n4))
    ax_d4.set_yticks(range(n4))
    ax_d4.set_xticklabels([f"P{j+1}" for j in range(n4)], fontsize=9)
    ax_d4.set_yticklabels([f"P{j+1}" for j in range(n4)], fontsize=9)
    ax_d4.set_title("Diversion Ratios (logit, four-product baseline)")
    ax_d4.set_xlabel("Product gaining the sale")
    ax_d4.set_ylabel("Product losing the sale")
    for i in range(n4):
        for j in range(n4):
            label = "" if i == j else f"{div4[i, j]:.2f}"
            ax_d4.text(j, i, label, ha="center", va="center", fontsize=10)
    cbar = plt.colorbar(im, ax=ax_d4)
    cbar.set_label("Diversion share")
    report.add_figure(
        "figures/four-product-diversion.png",
        "Diversion ratios between the four baseline products",
        fig_d4,
        description="Each row is the product losing a marginal sale. Each column is the "
        "product that absorbs it. Under logit, larger-share products absorb more diverted "
        "sales from every other product. The merger changes pricing through this matrix.",
    )

    table4 = {
        "Scenario": [],
        "Avg Price": [],
        "Price Change (%)": [],
        "Inside Share": [],
        "Outside Share": [],
        "FOC Residual": [],
    }
    for name, outcome in scenarios4.items():
        ps = outcome["prices"]
        ss = outcome["shares"]
        omegas = ownership_matrix(outcome["owners"])
        residual = np.max(np.abs(foc_logit(ps, outcome["mc"], alpha4, xi4, omegas)))
        table4["Scenario"].append(name)
        table4["Avg Price"].append(f"{np.mean(ps):.4f}")
        table4["Price Change (%)"].append(f"{100 * (np.mean(ps) / np.mean(p4) - 1):.2f}")
        table4["Inside Share"].append(f"{np.sum(ss):.4f}")
        table4["Outside Share"].append(f"{1.0 - np.sum(ss):.4f}")
        table4["FOC Residual"].append(f"{residual:.1e}")
    df_baseline = pd.DataFrame(table4)
    report.add_table(
        "tables/four-product-results.csv",
        "Four-product baseline scenarios",
        df_baseline,
        description="Average prices and inside share by scenario. The FOC residual "
        "confirms the reported prices solve the post-merger pricing equations. The 10% "
        "cost cut pulls prices below the unconstrained merger but leaves them above the "
        "pre-merger level.",
    )

    # ---------- Part 3 results: six-product extension ----------
    report.add_results(
        "### Part 3: Six-product extension\n\n"
        "Now we run the same merger logic on a market with six products and three demand "
        "systems. Each system matches the same observed shares and prices; linear and "
        "log-linear also match the full margin vector, and logit matches margins as closely "
        "as one price coefficient allows. So any "
        "differences below come from demand curvature, not from data. Firm 1 buys Firm 2. "
        "Products 1 through 4 move under common ownership."
    )

    demand_labels = ["Logit", "Linear", "Log-linear"]
    post_prices_all = [p_post_logit, p_post_linear, p_post_loglinear]
    fig_p6, axes_p6 = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors_pre_post = ["steelblue", "coral"]
    x6 = np.arange(J)
    bar_w = 0.35
    for ax, label, p_post in zip(axes_p6, demand_labels, post_prices_all):
        ax.bar(x6 - bar_w / 2, prices_obs, bar_w, label="Pre-merger", color=colors_pre_post[0])
        ax.bar(x6 + bar_w / 2, p_post, bar_w, label="Post-merger", color=colors_pre_post[1])
        ax.set_xlabel("Product")
        ax.set_title(f"{label} Demand")
        ax.set_xticks(x6)
        ax.set_xticklabels(product_names, fontsize=8, rotation=45)
        ax.legend(fontsize=8)
        for j in range(4):
            pct = (p_post[j] - prices_obs[j]) / prices_obs[j] * 100
            ax.annotate(f"+{pct:.1f}%", xy=(j + bar_w / 2, p_post[j]),
                        fontsize=7, ha="center", va="bottom", color="red")
    axes_p6[0].set_ylabel("Price")
    fig_p6.suptitle("Six-Product Extension: Pre- and Post-Merger Prices by Demand System",
                    fontsize=14, y=1.02)
    fig_p6.tight_layout()
    report.add_figure(
        "figures/price-comparison.png",
        "Pre- and post-merger prices for three demand systems (six products)",
        fig_p6,
        description="Logit and log-linear demand give double-digit average price increases "
        "on the merging products. Linear demand is much more muted under this cross-slope "
        "calibration. The same calibration anchors each system. The gap between them is "
        "curvature, not data.",
    )

    fig_w, ax_w = plt.subplots(figsize=(9, 5))
    x_w = np.arange(len(demand_labels))
    bar_w_w = 0.25
    cs_vals = [dCS[m] for m in demand_labels]
    ps_vals = [dPS[m] for m in demand_labels]
    w_vals = [dW[m] for m in demand_labels]
    ax_w.bar(x_w - bar_w_w, cs_vals, bar_w_w, label="$\\Delta$ CS", color="steelblue")
    ax_w.bar(x_w, ps_vals, bar_w_w, label="$\\Delta$ PS", color="coral")
    ax_w.bar(x_w + bar_w_w, w_vals, bar_w_w, label="$\\Delta$ W (total)", color="seagreen")
    ax_w.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax_w.set_xlabel("Demand Model")
    ax_w.set_ylabel("Welfare Change")
    ax_w.set_title("Welfare Decomposition: Consumer, Producer, and Total Surplus")
    ax_w.set_xticks(x_w)
    ax_w.set_xticklabels(demand_labels)
    ax_w.legend()
    report.add_figure(
        "figures/welfare-decomposition.png",
        "Welfare changes for consumers, producers, and total surplus",
        fig_w,
        description="The welfare bars separate consumer surplus, producer surplus, and "
        "their sum. Consumers lose under every demand system. Producer gains do not offset "
        "consumer losses here. The total welfare effect is negative across the board.",
    )

    fig_g, (ax_g_a, ax_g_b) = plt.subplots(1, 2, figsize=(14, 5))
    model_colors = ["steelblue", "coral", "seagreen"]
    dp_by_model = {"Logit": dp_logit, "Linear": dp_linear, "Log-linear": dp_loglinear}
    guppi_by_model = {"Logit": guppi_logit, "Linear": guppi_linear, "Log-linear": guppi_loglinear}
    upp_by_model = {"Logit": upp_logit, "Linear": upp_linear, "Log-linear": upp_loglinear}
    x_g = np.arange(len(demand_labels))
    bar_w_g = 0.35
    avg_guppi = [np.mean(guppi_by_model[label][:4]) * 100 for label in demand_labels]
    avg_actual = [np.mean(dp_by_model[label][:4]) for label in demand_labels]
    ax_g_a.bar(x_g - bar_w_g / 2, avg_guppi, bar_w_g, label="GUPPI screen", color="gray")
    ax_g_a.bar(x_g + bar_w_g / 2, avg_actual, bar_w_g, label="Solved equilibrium", color="seagreen")
    ax_g_a.set_xlabel("Demand Model")
    ax_g_a.set_ylabel("Average Merging-Product Effect (%)")
    ax_g_a.set_title("Screen vs Solved Price Increase")
    ax_g_a.set_xticks(x_g)
    ax_g_a.set_xticklabels(demand_labels)
    ax_g_a.legend(fontsize=9)
    ax_g_a.axhline(0, color="black", linewidth=0.8)
    xprod = np.arange(4)
    bar_w_upp = 0.25
    for idx, label in enumerate(demand_labels):
        ax_g_b.bar(
            xprod + (idx - 1) * bar_w_upp,
            upp_by_model[label][:4],
            bar_w_upp,
            label=label,
            color=model_colors[idx],
        )
    ax_g_b.set_xlabel("Product")
    ax_g_b.set_ylabel("UPP")
    ax_g_b.set_title("UPP for Newly Co-Owned Products")
    ax_g_b.set_xticks(xprod)
    ax_g_b.set_xticklabels(product_names[:4], fontsize=9)
    ax_g_b.legend(fontsize=9)
    ax_g_b.axhline(0, color="black", linewidth=0.8)
    fig_g.tight_layout()
    report.add_figure(
        "figures/upp-guppi.png",
        "GUPPI screens, solved price effects, and product-level UPP",
        fig_g,
        description="GUPPI uses observed margins and diversion. It is computed before any "
        "counterfactual is solved. The left panel treats the solved price increase as the "
        "benchmark. The gap between the screen and the equilibrium reflects pass-through, "
        "demand curvature, and rival price responses.",
    )

    fig_e, ax_e = plt.subplots(figsize=(9, 6))
    efficiency_levels = np.linspace(0.0, 0.60, 121)
    mc_by_model = {"Logit": mc_logit, "Linear": mc_linear, "Log-linear": mc_loglinear}
    post_price_by_model = {"Logit": p_post_logit, "Linear": p_post_linear,
                           "Log-linear": p_post_loglinear}
    break_even_efficiency = {}
    for label, color in zip(demand_labels, model_colors):
        avg_price_changes = []
        mc_base = mc_by_model[label]
        p_warm = post_price_by_model[label].copy()
        for eff in efficiency_levels:
            mc_eff = mc_base.copy()
            mc_eff[:4] = mc_base[:4] * (1.0 - eff)
            if label == "Logit":
                p_eff = scipy.optimize.fsolve(
                    foc_logit, x0=p_warm,
                    args=(mc_eff, cal_logit["alpha"], cal_logit["xi"], omega_post),
                )
            elif label == "Linear":
                p_eff = scipy.optimize.fsolve(
                    foc_linear, x0=p_warm,
                    args=(mc_eff, cal_linear["a"], cal_linear["B"], omega_post),
                )
            else:
                logp_eff = scipy.optimize.fsolve(
                    foc_loglinear_logp, x0=np.log(np.maximum(p_warm, 0.01)),
                    args=(mc_eff, cal_loglinear["a_ll"], cal_loglinear["E"], omega_post),
                )
                p_eff = np.exp(logp_eff)
            p_warm = p_eff.copy()
            avg_dp = np.mean((p_eff[:4] - prices_obs[:4]) / prices_obs[:4]) * 100
            avg_price_changes.append(avg_dp)
        efficiency_pct = efficiency_levels * 100
        break_even_efficiency[label] = first_zero_crossing(efficiency_pct, avg_price_changes)
        ax_e.plot(efficiency_pct, avg_price_changes, "o-", label=label,
                  color=color, markersize=4)
        if np.isfinite(break_even_efficiency[label]):
            ax_e.scatter(
                [break_even_efficiency[label]], [0.0],
                color=color, edgecolor="black", linewidth=0.6, zorder=5,
            )
    ax_e.axhline(0, color="black", linewidth=1.0, linestyle="--")
    ax_e.set_xlabel("Marginal Cost Reduction for Merging Firms (%)")
    ax_e.set_ylabel("Avg Price Change for Merging Products (%)")
    ax_e.set_title("Price Effects vs Efficiency Gains Frontier")
    ax_e.legend()
    ax_e.annotate("Price-increasing\nmerger", xy=(2, 2), fontsize=9, color="red",
                  ha="left", va="bottom")
    ax_e.annotate("Consumer-beneficial\nmerger", xy=(20, -2), fontsize=9, color="green",
                  ha="right", va="top")
    report.add_figure(
        "figures/efficiency-frontier.png",
        "Cost reductions needed to offset merger price increases",
        fig_e,
        description="The efficiency curve lowers marginal costs on products 1-4. Each "
        "point re-solves the post-merger pricing problem. The marker on the zero line is "
        "the break-even cost cut. Below the line, efficiencies reverse the average price "
        "increase. Above the line, the merger still raises prices.",
    )

    table_extended = {
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
    for label, p_post, guppi_vals, cmcr_vals in [
        ("Logit", p_post_logit, guppi_logit, cmcr_logit),
        ("Linear", p_post_linear, guppi_linear, cmcr_linear),
        ("Log-linear", p_post_loglinear, guppi_loglinear, cmcr_loglinear),
    ]:
        dp = (p_post - prices_obs) / prices_obs * 100
        avg_dp = np.mean(dp[:4])
        avg_guppi_val = np.mean(guppi_vals[:4]) * 100
        table_extended["Demand Model"].append(label)
        table_extended["Avg Actual Price Inc. (%)"].append(round(avg_dp, 2))
        table_extended["Max Price Change (%)"].append(round(np.max(dp[:4]), 2))
        table_extended["Avg GUPPI Screen (%)"].append(round(avg_guppi_val, 2))
        table_extended["Screen Gap (pp)"].append(round(avg_dp - avg_guppi_val, 2))
        table_extended["Avg CMCR Screen (%)"].append(round(np.mean(cmcr_vals[:4]) * 100, 2))
        table_extended["Break-even Eff. (%)"].append(round(break_even_efficiency[label], 2))
        table_extended["Delta CS"].append(round(dCS[label], 4))
        table_extended["Delta PS"].append(round(dPS[label], 4))
        table_extended["Delta W"].append(round(dW[label], 4))
        table_extended["Post FOC Residual"].append(f"{post_residuals[label]:.1e}")
    df_extended = pd.DataFrame(table_extended)
    report.add_table(
        "tables/merger-effects.csv",
        "Six-product merger price effects and screens",
        df_extended,
        description="The table puts local screens and solved counterfactuals side by side. "
        "Average price increases come from the post-merger FOC solution. GUPPI and CMCR "
        "are local screens at observed prices. Break-even efficiencies come from repeated "
        "FOC solves over a cost-reduction grid.",
    )

    # =====================================================================
    # Takeaway
    # =====================================================================
    report.add_takeaway(
        "Three tools answer three different questions about the same merger. "
        "Concentration screens are arithmetic. They triage candidate mergers straight from "
        "a sales table. A calibrated four-product Bertrand model turns one observed margin "
        "into a post-merger price prediction. The six-product extension shows that the "
        "same observed market admits different counterfactuals once you change the demand "
        "system.\n\n"
        "UPP, GUPPI, and CMCR are local screens at observed prices. They correlate with "
        "the solved equilibrium but do not match it. The gap comes from pass-through, "
        "rival reactions, and demand curvature. The efficiency frontier traces the "
        "marginal-cost cut that would offset the price effect under each demand system. "
        "The cut is large under logit and log-linear demand. It is small under linear "
        "demand in this calibration.\n\n"
        "An antitrust review that uses all three layers stays honest. A high delta-HHI "
        "does not pin down a price effect. A single demand system does not pin down "
        "robustness. Stacking the layers makes both the screen and its limits visible."
    )

    report.add_references([
        "Werden, G. and Froeb, L. (1994). \"The Effects of Mergers in Differentiated Products "
        "Industries: Logit Demand and Merger Policy.\" *Journal of Law, Economics, & Organization*, 10(2).",
        "Farrell, J. and Shapiro, C. (2010). \"Antitrust Evaluation of Horizontal Mergers: "
        "An Economic Alternative to Market Definition.\" *The B.E. Journal of Theoretical Economics*, 10(1).",
        "Berry, S. (1994). \"Estimating Discrete-Choice Models of Product Differentiation.\" "
        "*RAND Journal of Economics*, 25(2).",
        "U.S. Department of Justice and Federal Trade Commission (2023). *Merger Guidelines*.",
        "Tirole, J. (1988). *The Theory of Industrial Organization*. MIT Press, Ch. 5.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
