#!/usr/bin/env python3
"""AR processes: stationarity, estimation, and persistence.

Simulates AR(1) sample paths, estimates AR coefficients by OLS, computes
analytic and sample ACF, spectral density, and a convergence panel showing
how the OLS estimate of rho shrinks toward the truth as T grows.  A
multiplier-accelerator application shows how AR(1) fiscal-shock persistence
maps into macro dynamics.

References:
  Box, G. E. P. and Jenkins, G. M. (1970). Time Series Analysis: Forecasting
    and Control. Holden-Day.
  Hamilton, J. D. (1994). Time Series Analysis. Princeton University Press.
  Samuelson, P. A. (1939). Interactions Between the Multiplier Analysis and
    the Principle of Acceleration. Review of Economics and Statistics, 21(2).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


# ---------------------------------------------------------------------------
# AR(1) simulation and exact moments
# ---------------------------------------------------------------------------

def simulate_ar1(
    rho: float,
    sigma: float,
    periods: int,
    seed: int = 42,
    burn_in: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate x_t = rho x_{t-1} + eps_t and drop the burn-in transient."""
    rng = np.random.default_rng(seed)
    total_periods = periods + burn_in
    shocks = rng.normal(0.0, sigma, total_periods)
    x = np.zeros(total_periods)
    for t in range(1, total_periods):
        x[t] = rho * x[t - 1] + shocks[t]
    return x[burn_in:], shocks[burn_in:]


def irf_ar1(rho: float, periods: int) -> np.ndarray:
    """Exact AR(1) response to a unit innovation at date 0: rho^h."""
    return rho ** np.arange(periods)


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation normalized by the lag-zero variance."""
    centered = x - np.mean(x)
    denom = np.dot(centered, centered)
    acf = np.zeros(max_lag + 1)
    if denom < 1e-15:
        return acf
    for lag in range(max_lag + 1):
        acf[lag] = np.dot(centered[: -lag or None], centered[lag:]) / denom
    return acf


def spectral_density_ar1(
    rho: float, sigma: float, frequencies: np.ndarray
) -> np.ndarray:
    """Exact spectral density of x_t = rho x_{t-1} + eps_t."""
    denominator = np.abs(1.0 - rho * np.exp(-1j * frequencies)) ** 2
    return sigma**2 / (2.0 * np.pi * denominator)


# ---------------------------------------------------------------------------
# OLS estimation of AR(p)
# ---------------------------------------------------------------------------

def estimate_ar_ols(y: np.ndarray, p: int) -> tuple[np.ndarray, float]:
    """OLS estimate of AR(p) on a mean-zero series.

    Builds the (T-p) x p lagged design matrix, solves the normal equations,
    and returns coefficient vector phi_hat and residual variance sigma_hat^2.
    """
    T = len(y)
    # Design matrix: column k contains y_{t-k-1} for t = p+1, ..., T.
    X = np.column_stack([y[p - 1 - k : T - 1 - k] for k in range(p)])
    y_dep = y[p:]
    # phi_hat = (X'X)^{-1} X'y
    phi_hat = np.linalg.lstsq(X, y_dep, rcond=None)[0]
    residuals = y_dep - X @ phi_hat
    sigma_hat2 = float(np.dot(residuals, residuals) / (T - p))
    return phi_hat, sigma_hat2


def ols_convergence(
    rho_true: float,
    sigma: float,
    sample_sizes: list[int],
    n_reps: int = 500,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo mean and s.d. of rho_hat as a function of T.

    For each T, draws n_reps independent AR(1) paths and records the OLS
    estimate of rho.  Returns (mean_rho_hat, std_rho_hat) arrays over T.
    """
    rng = np.random.default_rng(seed)
    mean_hat = np.empty(len(sample_sizes))
    std_hat = np.empty(len(sample_sizes))
    for i, T in enumerate(sample_sizes):
        estimates = np.empty(n_reps)
        for rep in range(n_reps):
            # Simulate T + 200 steps, discard burn-in.
            eps = rng.normal(0.0, sigma, T + 200)
            y = np.zeros(T + 200)
            for t in range(1, T + 200):
                y[t] = rho_true * y[t - 1] + eps[t]
            y = y[200:]
            phi_hat, _ = estimate_ar_ols(y, p=1)
            estimates[rep] = phi_hat[0]
        mean_hat[i] = estimates.mean()
        std_hat[i] = estimates.std()
    return mean_hat, std_hat


# ---------------------------------------------------------------------------
# Multiplier-accelerator
# ---------------------------------------------------------------------------

def simulate_multiplier_accelerator(
    alpha: float,
    beta: float,
    rho_g: float,
    sigma: float,
    periods: int,
    seed: int = 43,
    burn_in: int = 200,
) -> dict[str, np.ndarray]:
    """Simulate deviations from the multiplier-accelerator steady state."""
    rng = np.random.default_rng(seed)
    total = periods + burn_in
    shocks = rng.normal(0.0, sigma, total)
    y = np.zeros(total)
    c = np.zeros(total)
    investment = np.zeros(total)
    g = np.zeros(total)
    for t in range(1, total):
        c[t] = beta * y[t - 1]
        g[t] = rho_g * g[t - 1] + shocks[t]
        investment[t] = alpha * (c[t] - c[t - 1])
        y[t] = c[t] + investment[t] + g[t]
    sl = slice(burn_in, None)
    return {"Y": y[sl], "C": c[sl], "I": investment[sl], "G": g[sl]}


def irf_multiplier_accelerator(
    alpha: float, beta: float, rho_g: float, periods: int
) -> dict[str, np.ndarray]:
    """Impulse response to a one-unit government-spending innovation."""
    y = np.zeros(periods)
    c = np.zeros(periods)
    investment = np.zeros(periods)
    g = np.zeros(periods)
    shocks = np.zeros(periods)
    shocks[0] = 1.0
    for t in range(periods):
        c[t] = beta * (y[t - 1] if t > 0 else 0.0)
        g[t] = rho_g * (g[t - 1] if t > 0 else 0.0) + shocks[t]
        c_lag = c[t - 1] if t > 0 else 0.0
        investment[t] = alpha * (c[t] - c_lag)
        y[t] = c[t] + investment[t] + g[t]
    return {"Y": y, "C": c, "I": investment, "G": g}


def format_roots(roots: np.ndarray) -> str:
    if np.all(np.abs(np.imag(roots)) < 1e-10):
        ordered = sorted(float(np.real(r)) for r in roots)
        return ", ".join(f"{r:.3f}" for r in ordered)
    return ", ".join(f"{r.real:.3f}{r.imag:+.3f}i" for r in roots)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tutorial_dir = Path(__file__).resolve().parent
    os.chdir(tutorial_dir)

    # AR(1) calibration.
    rho_ar1 = 0.9
    sigma_ar1 = 0.01

    # Multiplier-accelerator calibration.
    alpha_ma = 0.3
    beta_ma = 0.8
    rho_g = 0.9
    sigma_g = 0.01

    periods_sim = 220
    periods_irf = 40
    max_lag = 20

    print("Simulating AR(1) path...")
    ar1_path, _ = simulate_ar1(rho_ar1, sigma_ar1, periods_sim)
    ar1_variance = sigma_ar1**2 / (1.0 - rho_ar1**2)
    ar1_sd = float(np.sqrt(ar1_variance))
    ar1_half_life = np.log(0.5) / np.log(rho_ar1)

    # Population and sample ACF.
    ar1_acf = autocorrelation(ar1_path, max_lag=max_lag)
    ar1_acf_theory = rho_ar1 ** np.arange(max_lag + 1)

    # Spectral density for three persistence levels.
    frequencies = np.linspace(0.01, np.pi, 500)

    # OLS convergence: rho_hat vs T.
    print("Running OLS convergence Monte Carlo (500 reps x 6 sample sizes)...")
    sample_sizes = [50, 100, 200, 500, 2000, 10000]
    mean_hat, std_hat = ols_convergence(
        rho_ar1, sigma_ar1, sample_sizes, n_reps=500
    )

    # Multiplier-accelerator.
    ma_irf = irf_multiplier_accelerator(alpha_ma, beta_ma, rho_g, periods_irf)
    ma_roots = np.roots([1.0, -beta_ma * (1.0 + alpha_ma), alpha_ma * beta_ma])
    ma_root_text = format_roots(ma_roots)
    ma_root_modulus = float(np.max(np.abs(ma_roots)))

    setup_style()
    Path("figures").mkdir(exist_ok=True)

    # =========================================================================
    # Figure 1 (2x2): sample path | ACF (top), spectral density | OLS convergence (bottom)
    # =========================================================================
    fig_diag, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    ax_path, ax_acf = axes[0, 0], axes[0, 1]
    ax_spec, ax_conv = axes[1, 0], axes[1, 1]

    # Top-left: simulated path with ±2 s.d. band.
    sim_periods = np.arange(periods_sim)
    ax_path.plot(sim_periods, ar1_path, color="#2c7fb8", linewidth=1.1)
    ax_path.fill_between(
        sim_periods, -2.0 * ar1_sd, 2.0 * ar1_sd,
        color="#2c7fb8", alpha=0.12, label="Population $\\pm 2$ s.d.",
    )
    ax_path.axhline(0.0, color="black", linewidth=0.5)
    ax_path.set_xlabel("Period $t$")
    ax_path.set_ylabel("$x_t$")
    ax_path.set_title(f"AR(1) sample path ($\\rho = {rho_ar1}$)")
    ax_path.legend(loc="upper right", fontsize=9)

    # Top-right: sample ACF vs population rho^k.
    lags = np.arange(max_lag + 1)
    ax_acf.bar(lags, ar1_acf, color="#2c7fb8", alpha=0.55, label="Sample ACF")
    ax_acf.plot(lags, ar1_acf_theory, color="#b2182b", marker="o", markersize=4,
                linewidth=1.6, label="Population $\\rho^k$")
    ax_acf.axhline(0.0, color="black", linewidth=0.5)
    ax_acf.set_xlabel("Lag $k$")
    ax_acf.set_ylabel("Autocorrelation")
    ax_acf.set_title("Sample vs. population ACF")
    ax_acf.legend(fontsize=9)

    # Bottom-left: spectral density for three rho values.
    for rho_val, color in zip([0.5, 0.9, 0.99], ["#7fcdbb", "#2c7fb8", "#253494"]):
        spectrum = spectral_density_ar1(rho_val, sigma_ar1, frequencies)
        ax_spec.plot(frequencies / np.pi, spectrum, color=color,
                     linewidth=2.0, label=f"$\\rho = {rho_val}$")
    ax_spec.set_yscale("log")
    ax_spec.set_xlabel("Frequency $\\omega / \\pi$")
    ax_spec.set_ylabel("Spectral density $S_x(\\omega)$ (log)")
    ax_spec.set_title("Persistence loads variance at low frequencies")
    ax_spec.legend(fontsize=9)

    # Bottom-right: OLS rho_hat vs T with ±1 s.d. band — convergence to truth.
    T_arr = np.array(sample_sizes, dtype=float)
    ax_conv.plot(T_arr, mean_hat, color="tab:purple", linewidth=2.0,
                 marker="o", markersize=5, label="Mean $\\hat\\rho$ (OLS)")
    ax_conv.fill_between(
        T_arr, mean_hat - std_hat, mean_hat + std_hat,
        color="tab:purple", alpha=0.20, label="$\\pm 1$ s.d. across reps",
    )
    ax_conv.axhline(rho_ar1, color="0.4", linestyle="--", linewidth=1.0,
                    label=f"True $\\rho = {rho_ar1}$")
    ax_conv.set_xscale("log")
    ax_conv.set_xlabel("Sample size $T$ (log scale)")
    ax_conv.set_ylabel("$\\hat\\rho$")
    ax_conv.set_title("OLS converges to truth: $\\hat\\rho \\to \\rho$ as $T \\to \\infty$")
    ax_conv.legend(fontsize=9)

    fig_diag.tight_layout()
    save_figure(fig_diag, "figures/ar-diagnostics.png", dpi=150)

    # =========================================================================
    # Figure 2 (2x2): multiplier-accelerator impulse responses
    # =========================================================================
    periods_arr = np.arange(periods_irf)
    fig_ma, axes_ma = plt.subplots(2, 2, figsize=(12.0, 8.0))
    ma_series = [
        ("Y", "Income deviation $y_t$", "#2c7fb8"),
        ("C", "Consumption deviation $c_t$", "#d95f0e"),
        ("I", "Investment deviation $i_t$", "#7570b3"),
        ("G", "Government spending deviation $g_t$", "#238b45"),
    ]
    for ax, (key, title, color) in zip(axes_ma.flat, ma_series):
        ax.plot(periods_arr, ma_irf[key], color=color, linewidth=2.2)
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_xlabel("Periods after innovation")
        ax.set_ylabel("Deviation")
        ax.set_title(title)
    fig_ma.suptitle(
        "Multiplier-Accelerator Response to a Government Spending Innovation",
        fontsize=13,
    )
    fig_ma.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig_ma, "figures/multiplier-accelerator-irfs.png", dpi=150)

    # Thumbnail: use the diagnostics figure.
    save_thumbnail("figures/ar-diagnostics.png", "figures/thumb.png")

    # =========================================================================
    # Summary table
    # =========================================================================
    ar_summary = pd.DataFrame(
        {
            "Object": [
                "Persistence ($\\rho$)",
                "Unconditional variance",
                "Half-life (periods)",
                "First-order autocorrelation",
                "Spectral peak frequency",
            ],
            "$\\rho=0.5$": [
                "0.50",
                f"{sigma_ar1**2 / (1.0 - 0.5**2):.6f}",
                f"{np.log(0.5) / np.log(0.5):.1f}",
                "0.50",
                "0",
            ],
            "$\\rho=0.9$": [
                "0.90",
                f"{sigma_ar1**2 / (1.0 - 0.9**2):.6f}",
                f"{np.log(0.5) / np.log(0.9):.1f}",
                "0.90",
                "0",
            ],
            "$\\rho=0.99$": [
                "0.99",
                f"{sigma_ar1**2 / (1.0 - 0.99**2):.6f}",
                f"{np.log(0.5) / np.log(0.99):.1f}",
                "0.99",
                "0",
            ],
        }
    )
    Path("tables").mkdir(parents=True, exist_ok=True)
    ar_summary.to_csv("tables/ar-properties.csv", index=False)

    print(
        f"Generated 2 figures and 1 table.\n"
        f"  AR(1) half-life at rho={rho_ar1}: {ar1_half_life:.1f} periods\n"
        f"  MA roots: {ma_root_text}, modulus={ma_root_modulus:.3f}"
    )


if __name__ == "__main__":
    main()
