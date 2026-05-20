#!/usr/bin/env python3
"""Fiscal-shock persistence and multiplier-accelerator income dynamics."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


def simulate_ar1(
    rho: float,
    sigma: float,
    periods: int,
    seed: int = 42,
    burn_in: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate x_t = rho x_{t-1} + eps_t and drop the transient."""
    rng = np.random.default_rng(seed)
    total_periods = periods + burn_in
    shocks = rng.normal(0.0, sigma, total_periods)
    x = np.zeros(total_periods)

    for t in range(1, total_periods):
        x[t] = rho * x[t - 1] + shocks[t]

    return x[burn_in:], shocks[burn_in:]


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
    total_periods = periods + burn_in
    shocks = rng.normal(0.0, sigma, total_periods)

    y = np.zeros(total_periods)
    c = np.zeros(total_periods)
    investment = np.zeros(total_periods)
    g = np.zeros(total_periods)

    for t in range(1, total_periods):
        c[t] = beta * y[t - 1]
        g[t] = rho_g * g[t - 1] + shocks[t]
        investment[t] = alpha * (c[t] - c[t - 1])
        y[t] = c[t] + investment[t] + g[t]

    sl = slice(burn_in, None)
    return {
        "Y": y[sl],
        "C": c[sl],
        "I": investment[sl],
        "G": g[sl],
        "eps": shocks[sl],
    }


def irf_ar1(rho: float, periods: int) -> np.ndarray:
    """Exact AR(1) response to a unit innovation at date 0."""
    return rho ** np.arange(periods)


def irf_multiplier_accelerator(
    alpha: float,
    beta: float,
    rho_g: float,
    periods: int,
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


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation normalized by the lag-zero covariance."""
    centered = x - np.mean(x)
    denom = np.dot(centered, centered)
    acf = np.zeros(max_lag + 1)

    if denom < 1e-15:
        return acf

    for lag in range(max_lag + 1):
        acf[lag] = np.dot(centered[:-lag or None], centered[lag:]) / denom

    return acf


def spectral_density_ar1(rho: float, sigma: float, frequencies: np.ndarray) -> np.ndarray:
    """Exact spectral density of x_t = rho x_{t-1} + eps_t."""
    denominator = np.abs(1.0 - rho * np.exp(-1j * frequencies)) ** 2
    return sigma**2 / (2.0 * np.pi * denominator)


def format_roots(roots: np.ndarray) -> str:
    """Format characteristic roots for text output."""
    if np.all(np.abs(np.imag(roots)) < 1e-10):
        ordered = sorted(float(np.real(root)) for root in roots)
        return ", ".join(f"{root:.3f}" for root in ordered)
    return ", ".join(f"{root.real:.3f}{root.imag:+.3f}i" for root in roots)


def main() -> None:
    tutorial_dir = Path(__file__).resolve().parent
    os.chdir(tutorial_dir)

    # AR(1) calibration.
    rho_ar1 = 0.9
    sigma_ar1 = 0.01

    # Samuelson multiplier-accelerator calibration.
    alpha_ma = 0.3
    beta_ma = 0.8
    rho_g = 0.9
    sigma_g = 0.01
    g_bar = 1.0

    periods_sim = 220
    periods_irf = 40
    max_lag = 20

    print("Simulating fiscal-shock persistence and income dynamics...")
    ar1_path, _ = simulate_ar1(rho_ar1, sigma_ar1, periods_sim)
    ma_sim = simulate_multiplier_accelerator(
        alpha_ma,
        beta_ma,
        rho_g,
        sigma_g,
        periods_sim,
    )

    rho_values = [0.5, 0.7, 0.9, 0.99]
    ar1_irfs = {rho: irf_ar1(rho, periods_irf) for rho in rho_values}
    ma_irf = irf_multiplier_accelerator(alpha_ma, beta_ma, rho_g, periods_irf)

    ar1_acf = autocorrelation(ar1_path, max_lag=max_lag)
    ar1_acf_theory = rho_ar1 ** np.arange(max_lag + 1)
    ma_y_acf = autocorrelation(ma_sim["Y"], max_lag=max_lag)

    frequencies = np.linspace(0.01, np.pi, 500)
    ar1_variance = sigma_ar1**2 / (1.0 - rho_ar1**2)
    ar1_sd = np.sqrt(ar1_variance)
    ar1_half_life = np.log(0.5) / np.log(rho_ar1)

    y_bar = g_bar / (1.0 - beta_ma)
    c_bar = beta_ma * y_bar
    ma_roots = np.roots([1.0, -beta_ma * (1.0 + alpha_ma), alpha_ma * beta_ma])
    ma_root_text = format_roots(ma_roots)
    ma_root_modulus = np.max(np.abs(ma_roots))

    setup_style()

    periods = np.arange(periods_irf)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = ["#7fcdbb", "#41b6c4", "#2c7fb8", "#253494"]
    for (rho, irf), color in zip(ar1_irfs.items(), colors):
        ax1.plot(periods, irf, color=color, linewidth=2.5, label=f"$\\rho={rho}$")
    ax1.axhline(0.5, color="black", linewidth=0.8, linestyle=":", label="Half response")
    ax1.axhline(0.0, color="black", linewidth=0.5)
    ax1.set_xlabel("Periods after innovation")
    ax1.set_ylabel("Response of $x_t$")
    ax1.set_title("Persistence Sets the Horizon of a Unit Innovation")
    ax1.legend(fontsize=10)
    save_figure(fig1, "figures/ar1-irfs.png", dpi=150)

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    ma_series = [
        ("Y", "Income deviation $y_t$", "#2c7fb8"),
        ("C", "Consumption deviation $c_t$", "#d95f0e"),
        ("I", "Investment deviation $i_t$", "#7570b3"),
        ("G", "Government spending deviation $g_t$", "#238b45"),
    ]
    for ax, (key, title, color) in zip(axes2.flat, ma_series):
        ax.plot(periods, ma_irf[key], color=color, linewidth=2.4)
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_xlabel("Periods after innovation")
        ax.set_ylabel("Deviation")
        ax.set_title(title)
    fig2.suptitle("Multiplier-Accelerator Response to Government Spending", fontsize=14)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig2, "figures/multiplier-accelerator-irfs.png", dpi=150)

    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    sim_periods = np.arange(periods_sim)
    ax3a.plot(sim_periods, ar1_path, color="#2c7fb8", linewidth=1.1)
    ax3a.fill_between(
        sim_periods,
        -2.0 * ar1_sd,
        2.0 * ar1_sd,
        color="#2c7fb8",
        alpha=0.12,
        label="Population +/- 2 s.d.",
    )
    ax3a.axhline(0.0, color="black", linewidth=0.5)
    ax3a.set_ylabel("$x_t$")
    ax3a.set_title("Simulated Persistent State")
    ax3a.legend(loc="upper right", fontsize=9)

    ax3b.plot(sim_periods, ma_sim["Y"], color="#2c7fb8", linewidth=1.0, label="$y_t$")
    ax3b.plot(sim_periods, ma_sim["C"], color="#d95f0e", linewidth=1.0, label="$c_t$", alpha=0.85)
    ax3b.plot(sim_periods, ma_sim["G"], color="#238b45", linewidth=1.0, label="$g_t$", alpha=0.85)
    ax3b.axhline(0.0, color="black", linewidth=0.5)
    ax3b.set_xlabel("Period")
    ax3b.set_ylabel("Deviation from steady state")
    ax3b.set_title("Simulated Multiplier-Accelerator Economy")
    ax3b.legend(loc="upper right", fontsize=9)
    fig3.tight_layout()
    save_figure(fig3, "figures/simulated-paths.png", dpi=150)

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))
    lags = np.arange(max_lag + 1)
    ax4a.bar(lags, ar1_acf, color="#2c7fb8", alpha=0.6, label="Simulation")
    ax4a.plot(lags, ar1_acf_theory, color="#b2182b", marker="o", markersize=4,
              linewidth=1.6, label="Population $\\rho^k$")
    ax4a.axhline(0.0, color="black", linewidth=0.5)
    ax4a.set_xlabel("Lag $k$")
    ax4a.set_ylabel("Autocorrelation")
    ax4a.set_title("AR(1): Sample vs. Population ACF")
    ax4a.legend(fontsize=9)

    ax4b.bar(lags, ma_y_acf, color="#d95f0e", alpha=0.7)
    ax4b.axhline(0.0, color="black", linewidth=0.5)
    ax4b.set_xlabel("Lag $k$")
    ax4b.set_ylabel("Autocorrelation")
    ax4b.set_title("Multiplier-Accelerator Output ACF")
    fig4.tight_layout()
    save_figure(fig4, "figures/autocorrelation.png", dpi=150)

    fig5, ax5 = plt.subplots(figsize=(10, 6))
    for rho, color in zip([0.5, 0.9, 0.99], ["#7fcdbb", "#2c7fb8", "#253494"]):
        spectrum = spectral_density_ar1(rho, sigma_ar1, frequencies)
        ax5.plot(frequencies, spectrum, color=color, linewidth=2.2, label=f"$\\rho={rho}$")
    ax5.set_xlabel("Frequency $\\omega$")
    ax5.set_ylabel("Spectral density $S_x(\\omega)$")
    ax5.set_title("Persistent Shocks Put More Power at Low Frequencies")
    ax5.set_yscale("log")
    ax5.legend(fontsize=10)
    save_figure(fig5, "figures/spectral-density.png", dpi=150)

    ar_summary = pd.DataFrame(
        {
            "Object": [
                "Persistence ($\\rho$)",
                "Unconditional variance",
                "Half-life (periods)",
                "First-order autocorrelation",
                "Spectral peak",
            ],
            "$\\rho=0.5$": [
                "0.50",
                f"{sigma_ar1**2 / (1.0 - 0.5**2):.6f}",
                f"{np.log(0.5) / np.log(0.5):.1f}",
                "0.50",
                "Frequency 0",
            ],
            "$\\rho=0.9$": [
                "0.90",
                f"{sigma_ar1**2 / (1.0 - 0.9**2):.6f}",
                f"{np.log(0.5) / np.log(0.9):.1f}",
                "0.90",
                "Frequency 0",
            ],
            "$\\rho=0.99$": [
                "0.99",
                f"{sigma_ar1**2 / (1.0 - 0.99**2):.6f}",
                f"{np.log(0.5) / np.log(0.99):.1f}",
                "0.99",
                "Frequency 0",
            ],
        }
    )
    Path("tables").mkdir(parents=True, exist_ok=True)
    ar_summary.to_csv("tables/ar-properties.csv", index=False)

    save_thumbnail("figures/ar1-irfs.png", "figures/thumb.png")

    print(f"Generated 5 figures and 1 table.")
    print(f"  AR(1) half-life at rho=0.9: {ar1_half_life:.1f} periods")
    print(f"  MA roots: {ma_root_text}, modulus={ma_root_modulus:.3f}")


if __name__ == "__main__":
    main()
