#!/usr/bin/env python3
"""Estimating present bias from convex time budgets (Andreoni-Sprenger 2012).

Subjects allocate a token budget between a sooner payment date $t$ and a
later date $t + k$ at a posted gross interest rate $1 + r$. The
allocation is the solution to a standard intertemporal optimisation
under quasi-hyperbolic discounting. Two estimation methods recover the
present-bias parameter $\\beta$, the per-day discount factor $\\delta$,
and the CRRA curvature exponent $\\alpha$:

- Method 1: nonlinear least squares on the closed-form demand function
  (Andreoni-Sprenger eq 5).
- Method 2: two-limit Tobit maximum likelihood on the log-tangency
  linearisation (eq 6) with corner censoring.

A weak-vs-strong-design comparison shows why varying the front-end
delay $t$ is what separates $\\beta$ from $\\delta$.

References:
- Andreoni, J., & Sprenger, C. (2012). Estimating Time Preferences from
  Convex Budgets. American Economic Review 102(7), 3333-3356.
- Cohen, J., Ericson, K. M., Laibson, D., & White, J. M. (2020).
  Measuring Time Preferences. Journal of Economic Literature 58(2),
  299-347.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


# ---------------------------------------------------------------------------
# Model: closed-form demand with omega_1 = omega_2 = 0 (column-3 spec in AS)
# ---------------------------------------------------------------------------
def beta_eff(beta: float, t_arr: np.ndarray) -> np.ndarray:
    """Effective present-bias factor: beta when t == 0, else 1."""
    return np.where(t_arr == 0, beta, 1.0)


def tangency_ratio(beta: float, delta: float, alpha: float,
                   one_plus_r: np.ndarray, k: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Closed-form $c_t / c_{t+k}$ from Andreoni-Sprenger eq (4) with omega = 0."""
    be = beta_eff(beta, t)
    expo = 1.0 / (alpha - 1.0)
    return (be * delta**k * one_plus_r)**expo


def demand_c_t(beta: float, delta: float, alpha: float,
               one_plus_r: np.ndarray, k: np.ndarray, t: np.ndarray,
               m: np.ndarray) -> np.ndarray:
    """Closed-form sooner-payment demand (eq 5 with omega_1 = omega_2 = 0).

    Budget: (1 + r) c_t + c_{t+k} = m.
    Tangency: c_t / c_{t+k} = xi.
    Solve for c_t in dollar units.
    """
    xi = tangency_ratio(beta, delta, alpha, one_plus_r, k, t)
    return m * xi / (1.0 + one_plus_r * xi)


def log_tangency(beta: float, delta: float, alpha: float,
                 one_plus_r: np.ndarray, k: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Log of eq (4) with omega = 0. Linear in (1{t=0}, k, log(1+r))."""
    coef = 1.0 / (alpha - 1.0)
    return coef * (np.log(beta) * (t == 0).astype(float)
                   + k * np.log(delta)
                   + np.log(one_plus_r))


# ---------------------------------------------------------------------------
# Estimation Method 1: NLS on the demand function
# ---------------------------------------------------------------------------
def nls_residuals(theta: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Residuals of observed sooner earnings against eq-5 prediction."""
    beta, delta, alpha = theta
    pred = demand_c_t(beta, delta, alpha,
                      df["one_plus_r"].to_numpy(),
                      df["k"].to_numpy(),
                      df["t"].to_numpy(),
                      df["m"].to_numpy())
    return df["c_t"].to_numpy() - pred


def fit_nls(df: pd.DataFrame, theta0: np.ndarray) -> tuple:
    """Fit (beta, delta, alpha) by NLS on the demand function."""
    bounds_lo = np.array([0.50, 0.95, 0.10])
    bounds_hi = np.array([1.50, 1.00, 0.99])
    result = least_squares(
        nls_residuals, theta0, bounds=(bounds_lo, bounds_hi),
        args=(df,), method="trf", xtol=1e-10, ftol=1e-10,
    )
    return result.x, result


# ---------------------------------------------------------------------------
# Estimation Method 2: two-limit Tobit MLE on the log tangency
# ---------------------------------------------------------------------------
def tobit_neg_loglik(params: np.ndarray, df: pd.DataFrame) -> float:
    """Negative log-likelihood of two-limit Tobit on the log tangency.

    The log of c_t / c_{t+k} is linear in (1{t=0}, k, log(1+r)) with
    coefficients (a, b, c). Map back to (beta, delta, alpha) by:
        alpha = 1 + 1/c,   delta = exp(b/c),   beta = exp(a/c).
    Censoring rule: at the upper corner c_t/c_{t+k} -> infty as c_{t+k} -> 0,
    so observed log ratio above log(m / eps) is censored. Symmetrically
    at the lower corner. We use the observed corner indicators.
    """
    a_coef, b_coef, c_coef, sigma = params
    if sigma <= 0 or c_coef >= 0:
        return 1e10
    one_plus_r = df["one_plus_r"].to_numpy()
    k = df["k"].to_numpy()
    t = df["t"].to_numpy()
    y = df["log_ratio"].to_numpy()
    is_lower = df["censored_lower"].to_numpy()
    is_upper = df["censored_upper"].to_numpy()
    is_interior = ~(is_lower | is_upper)
    mu = a_coef * (t == 0).astype(float) + b_coef * k + c_coef * np.log(one_plus_r)
    z = (y - mu) / sigma
    log_lik = np.zeros_like(y)
    log_lik[is_interior] = norm.logpdf(z[is_interior]) - np.log(sigma)
    if is_lower.any():
        log_lik[is_lower] = norm.logcdf(z[is_lower])
    if is_upper.any():
        log_lik[is_upper] = norm.logsf(z[is_upper])
    return -float(np.sum(log_lik))


def fit_tobit(df: pd.DataFrame, theta0: np.ndarray) -> tuple:
    """Fit log-tangency coefficients (a, b, c, sigma) by ML and map back."""
    bounds = [(-2.0, 2.0), (-1e-2, 1e-2), (-50.0, -1e-3), (1e-4, 5.0)]
    res = minimize(tobit_neg_loglik, theta0, args=(df,),
                   method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500, "ftol": 1e-10})
    a_hat, b_hat, c_hat, sigma_hat = res.x
    alpha_hat = 1.0 + 1.0 / c_hat
    delta_hat = float(np.exp(b_hat / c_hat))
    beta_hat = float(np.exp(a_hat / c_hat))
    return (beta_hat, delta_hat, alpha_hat, sigma_hat), res


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def build_choice_sets() -> pd.DataFrame:
    """Replicate the AS 3x3 (t, k) design with five (a_t, a_{t+k}) cells each.

    Token budget is 100 in every choice. Sooner token rate a_t varies; later
    token rate a_{t+k} is 0.20 in the headline cells (one cell uses 0.25).
    Gross interest rate is 1 + r = a_{t+k} / a_t.
    """
    ts = [0, 7, 35]
    ks = [35, 70, 98]
    interest_rates = [1.05, 1.11, 1.25, 1.43, 2.00]
    rows = []
    for t_val in ts:
        for k_val in ks:
            for one_plus_r in interest_rates:
                a_late = 0.20
                a_early = a_late / one_plus_r
                m = 100.0 * a_late
                rows.append({
                    "t": t_val, "k": k_val,
                    "one_plus_r": one_plus_r,
                    "a_early": a_early, "a_late": a_late,
                    "m": m,
                })
    return pd.DataFrame(rows)


def simulate_subjects(choice_sets: pd.DataFrame, n_subjects: int,
                      beta_true: float, delta_true: float, alpha_true: float,
                      sigma_eps: float, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate $c_t$ for many subjects with iid normal noise on log tangency."""
    n_cells = len(choice_sets)
    expanded = pd.concat([choice_sets] * n_subjects, ignore_index=True)
    expanded["subject"] = np.repeat(np.arange(n_subjects), n_cells)
    log_ratio_true = log_tangency(beta_true, delta_true, alpha_true,
                                  expanded["one_plus_r"].to_numpy(),
                                  expanded["k"].to_numpy(),
                                  expanded["t"].to_numpy())
    eps = rng.normal(0.0, sigma_eps, size=len(expanded))
    log_ratio_obs = log_ratio_true + eps
    ratio_obs = np.exp(log_ratio_obs)
    one_plus_r = expanded["one_plus_r"].to_numpy()
    m = expanded["m"].to_numpy()
    c_t = m * ratio_obs / (1.0 + one_plus_r * ratio_obs)
    n_t = c_t / expanded["a_early"].to_numpy()
    n_t_clipped = np.clip(n_t, 0.0, 100.0)
    censored_lower = n_t < 0.5
    censored_upper = n_t > 99.5
    c_t_obs = n_t_clipped * expanded["a_early"].to_numpy()
    c_late_obs = (100.0 - n_t_clipped) * expanded["a_late"].to_numpy()
    eps_clip = 1e-6
    log_ratio_obs_safe = np.where(
        censored_lower | censored_upper,
        np.log(np.maximum(c_t_obs, eps_clip) / np.maximum(c_late_obs, eps_clip)),
        np.log(c_t_obs / c_late_obs),
    )
    expanded["c_t"] = c_t_obs
    expanded["c_late"] = c_late_obs
    expanded["n_t"] = n_t_clipped
    expanded["log_ratio"] = log_ratio_obs_safe
    expanded["censored_lower"] = censored_lower
    expanded["censored_upper"] = censored_upper
    return expanded


def bootstrap_se(df: pd.DataFrame, fit_fn, theta0: np.ndarray,
                 n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Subject-cluster bootstrap standard errors for the parameter estimates."""
    subjects = df["subject"].unique()
    estimates = []
    for _ in range(n_boot):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        boot_df = pd.concat([df[df["subject"] == s] for s in sampled],
                            ignore_index=True)
        try:
            theta_hat, _ = fit_fn(boot_df, theta0)
            estimates.append(theta_hat[:3])
        except Exception:
            continue
    estimates = np.array(estimates)
    if len(estimates) == 0:
        return np.full(3, np.nan)
    return estimates.std(axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(20260509)

    # True parameters: deliberately set beta < 1 so the design must detect bias.
    beta_true = 0.90
    delta_true = 0.99928   # daily; (1 / delta)^365 - 1 approx 0.30 annual
    alpha_true = 0.92      # CRRA exponent in u(c) = c^alpha / alpha
    sigma_eps = 0.30       # noise std on log tangency
    n_subjects = 100
    n_boot = 200
    annual_rate_true = (1.0 / delta_true) ** 365 - 1.0

    # Design grid
    choice_sets = build_choice_sets()
    full_design = simulate_subjects(choice_sets, n_subjects,
                                    beta_true, delta_true, alpha_true,
                                    sigma_eps, rng)
    weak_design = full_design[full_design["t"] == 0].copy()

    # Method 1: NLS on the demand function
    theta0_nls = np.array([1.00, 0.999, 0.95])
    nls_full, _ = fit_nls(full_design, theta0_nls)
    nls_weak, _ = fit_nls(weak_design, theta0_nls)
    nls_full_se = bootstrap_se(full_design, fit_nls, theta0_nls, n_boot, rng)
    nls_weak_se = bootstrap_se(weak_design, fit_nls, theta0_nls, n_boot, rng)

    # Method 2: two-limit Tobit MLE on log tangency
    # Initial coefficients: rough OLS on interior
    interior = full_design[~(full_design["censored_lower"] | full_design["censored_upper"])]
    X = np.column_stack([
        (interior["t"].to_numpy() == 0).astype(float),
        interior["k"].to_numpy(),
        np.log(interior["one_plus_r"].to_numpy()),
    ])
    y = interior["log_ratio"].to_numpy()
    ols_coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    sigma0 = float(np.std(y - X @ ols_coef))
    theta0_tobit = np.array([ols_coef[0], ols_coef[1], ols_coef[2], sigma0])
    tobit_full, _ = fit_tobit(full_design, theta0_tobit)
    beta_tobit, delta_tobit, alpha_tobit, _ = tobit_full

    # Identification figure: profile log-likelihood for beta
    beta_grid = np.linspace(0.65, 1.20, 41)
    profile_full = np.zeros_like(beta_grid)
    profile_weak = np.zeros_like(beta_grid)
    for i, b_try in enumerate(beta_grid):
        def neg_ll_concentrated(params, df, b=b_try):
            delta_, alpha_, sig_ = params
            if sig_ <= 0 or alpha_ >= 1.0 or alpha_ <= 0.05:
                return 1e10
            mu = log_tangency(b, delta_, alpha_,
                              df["one_plus_r"].to_numpy(),
                              df["k"].to_numpy(),
                              df["t"].to_numpy())
            resid = df["log_ratio"].to_numpy() - mu
            return 0.5 * len(resid) * np.log(2 * np.pi * sig_**2) \
                   + 0.5 * np.sum(resid**2) / sig_**2
        res_full = minimize(neg_ll_concentrated,
                            x0=[0.999, 0.95, 0.30], args=(full_design,),
                            method="L-BFGS-B",
                            bounds=[(0.95, 1.0), (0.10, 0.99), (1e-3, 5.0)])
        res_weak = minimize(neg_ll_concentrated,
                            x0=[0.999, 0.95, 0.30], args=(weak_design,),
                            method="L-BFGS-B",
                            bounds=[(0.95, 1.0), (0.10, 0.99), (1e-3, 5.0)])
        profile_full[i] = -res_full.fun
        profile_weak[i] = -res_weak.fun
    profile_full -= profile_full.max()
    profile_weak -= profile_weak.max()

    # =====================================================================
    # Figures and tables
    # =====================================================================
    setup_style()

    # ------------------------------------------------------------------
    # Figure 1: mean sooner tokens by interest rate, faceted by delay k
    # ------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    k_panels = [35, 70, 98]
    t_styles = {0: ("o", "tab:red"), 7: ("s", "tab:orange"), 35: ("^", "tab:blue")}
    for ax, k_val in zip(axes1, k_panels):
        for t_val, (marker, color) in t_styles.items():
            mask = (full_design["k"] == k_val) & (full_design["t"] == t_val)
            grouped = full_design.loc[mask].groupby("one_plus_r")["n_t"].mean()
            ax.plot(grouped.index, grouped.values, marker=marker, color=color,
                    linestyle="--", linewidth=1.0, markersize=6,
                    label=f"$t = {t_val}$ days")
        # Model fit at NLS estimates
        rs_smooth = np.linspace(1.0, 2.05, 60)
        for t_val, (_, color) in t_styles.items():
            ks_arr = np.full_like(rs_smooth, k_val)
            ts_arr = np.full_like(rs_smooth, t_val)
            ms_arr = np.full_like(rs_smooth, 100.0 * 0.20)
            xi = tangency_ratio(nls_full[0], nls_full[1], nls_full[2],
                                rs_smooth, ks_arr, ts_arr)
            ct_pred = ms_arr * xi / (1.0 + rs_smooth * xi)
            n_t_pred = ct_pred / (0.20 / rs_smooth)
            ax.plot(rs_smooth, n_t_pred, color=color, linestyle="-",
                    linewidth=1.2, alpha=0.6)
        ax.set_xlabel(r"Gross interest rate $1 + r$")
        ax.set_title(fr"$k = {k_val}$ days")
        ax.set_xlim(0.95, 2.10)
    axes1[0].set_ylabel("Mean sooner tokens (out of 100)")
    axes1[0].legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig1.tight_layout()
    save_figure(fig1, "figures/mean-allocations.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 2: identification - profile log-likelihood for beta
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(beta_grid, profile_full, color="tab:blue", linewidth=2,
             label="Strong design: $t \\in \\{0, 7, 35\\}$")
    ax2.plot(beta_grid, profile_weak, color="tab:orange", linewidth=2,
             linestyle="--", label="Weak design: $t = 0$ only")
    ax2.axvline(beta_true, color="tab:red", linestyle=":", linewidth=1.5,
                label=fr"True $\beta = {beta_true}$")
    ax2.axhline(-1.92, color="gray", linewidth=0.8, alpha=0.6,
                label="95% confidence level (chi-square 1 df)")
    ax2.set_xlabel(r"$\beta$")
    ax2.set_ylabel(r"Profiled Gaussian criterion (max-relative)")
    ax2.set_title("Identification of present bias under the two designs")
    ax2.legend(loc="lower center", fontsize=9)
    ax2.set_ylim(-30, 1)
    save_figure(fig2, "figures/identification-profile.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 3: NLS vs Tobit point estimates with bootstrap bands
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 3, figsize=(13, 4))
    param_names = [r"$\beta$", r"$\delta$", r"$\alpha$"]
    truths = [beta_true, delta_true, alpha_true]
    nls_est = nls_full
    tobit_est = (beta_tobit, delta_tobit, alpha_tobit)
    for ax, name, truth, nls_v, tobit_v, nls_se in zip(
        axes3, param_names, truths, nls_est, tobit_est, nls_full_se
    ):
        positions = [0, 1]
        ax.errorbar([0], [nls_v], yerr=[nls_se], fmt="o", color="tab:blue",
                    markersize=10, capsize=5, linewidth=1.5,
                    label="NLS bootstrap SE")
        ax.scatter([1], [tobit_v], marker="s", s=100, color="tab:green",
                   label="Tobit point")
        ax.axhline(truth, color="tab:red", linestyle="--", linewidth=1.0,
                   label=f"True = {truth:.4f}")
        ax.set_xticks(positions)
        ax.set_xticklabels(["NLS\n(Method 1)", "Tobit ML\n(Method 2)"])
        ax.set_ylabel(name)
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(name)
        ax.legend(fontsize=8, loc="best")
    fig3.tight_layout()
    save_figure(fig3, "figures/method-comparison.png", dpi=150)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    recovery_table = pd.DataFrame({
        "Parameter": ["Present bias beta", "Daily discount factor delta",
                       "CRRA exponent alpha"],
        "True": [f"{beta_true:.4f}", f"{delta_true:.5f}", f"{alpha_true:.4f}"],
        "NLS estimate": [f"{nls_full[0]:.4f}", f"{nls_full[1]:.5f}",
                          f"{nls_full[2]:.4f}"],
        "NLS bootstrap SE": [f"{nls_full_se[0]:.4f}", f"{nls_full_se[1]:.5f}",
                              f"{nls_full_se[2]:.4f}"],
        "Tobit estimate": [f"{beta_tobit:.4f}", f"{delta_tobit:.5f}",
                            f"{alpha_tobit:.4f}"],
    })
    Path("tables").mkdir(parents=True, exist_ok=True)
    recovery_table.to_csv("tables/parameter-recovery.csv", index=False)

    design_table = pd.DataFrame({
        "Design": ["Strong (t in 0, 7, 35)", "Weak (t = 0 only)"],
        "Cells per subject": [len(choice_sets), int((choice_sets["t"] == 0).sum())],
        "NLS beta estimate": [f"{nls_full[0]:.4f}", f"{nls_weak[0]:.4f}"],
        "NLS beta bootstrap SE": [f"{nls_full_se[0]:.4f}", f"{nls_weak_se[0]:.4f}"],
        "NLS delta estimate": [f"{nls_full[1]:.5f}", f"{nls_weak[1]:.5f}"],
        "NLS alpha estimate": [f"{nls_full[2]:.4f}", f"{nls_weak[2]:.4f}"],
    })
    design_table.to_csv("tables/design-comparison.csv", index=False)

    save_thumbnail("figures/mean-allocations.png", "figures/thumb.png")
    print("\nDone: 3 figures, 2 tables, thumb reproduced.")


if __name__ == "__main__":
    main()
