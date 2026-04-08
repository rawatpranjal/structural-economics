#!/usr/bin/env python3
"""AR Process Dynamics: AR(1) and Multiplier-Accelerator Models.

Parses the Dynare .mod files from ar1/ and ar2/, simulates the processes in
Python, and demonstrates impulse responses, simulated paths, autocorrelation
functions, and spectral densities.

Reference: Hamilton (1994), Time Series Analysis.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def parse_mod_file(mod_path: str) -> str:
    """Read a .mod file and return its contents."""
    return Path(mod_path).read_text()


def simulate_ar1(rho, sigma, T, seed=42):
    """Simulate an AR(1) process: x(t) = rho * x(t-1) + e(t)."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, T)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + e[t]
    return x, e


def simulate_multiplier_accelerator(alpha, beta_param, rho, sigma, Gbar, T, seed=42):
    """Simulate the multiplier-accelerator model from ar2/model.mod.

    C(t) = beta * Y(t-1)
    G(t) = rho * G(t-1) + (1-rho)*Gbar + e(t)
    I(t) = alpha * (C(t) - C(t-1))
    Y(t) = C(t) + I(t) + G(t)
    """
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, T)

    Y = np.zeros(T)
    C = np.zeros(T)
    I = np.zeros(T)
    G = np.zeros(T)

    for t in range(1, T):
        C[t] = beta_param * Y[t - 1]
        G[t] = rho * G[t - 1] + (1 - rho) * Gbar + e[t]
        c_prev = C[t - 1] if t > 0 else 0
        I[t] = alpha * (C[t] - c_prev)
        Y[t] = C[t] + I[t] + G[t]

    return {"Y": Y, "C": C, "I": I, "G": G, "e": e}


def irf_ar1(rho, T=40):
    """Analytical IRF for AR(1): rho^t."""
    return rho ** np.arange(T)


def irf_multiplier_accelerator(alpha, beta_param, rho, Gbar, T=40):
    """Compute IRF of the multiplier-accelerator model to a unit shock to G."""
    Y = np.zeros(T)
    C = np.zeros(T)
    I_arr = np.zeros(T)
    G = np.zeros(T)

    # Unit shock at t=0
    e = np.zeros(T)
    e[0] = 1.0

    for t in range(T):
        if t > 0:
            C[t] = beta_param * Y[t - 1]
        G[t] = rho * (G[t - 1] if t > 0 else 0) + e[t]
        c_prev = C[t - 1] if t > 0 else 0
        I_arr[t] = alpha * (C[t] - c_prev)
        Y[t] = C[t] + I_arr[t] + G[t]

    return {"Y": Y, "C": C, "I": I_arr, "G": G}


def autocorrelation(x, max_lag=20):
    """Compute sample autocorrelation function."""
    x_centered = x - np.mean(x)
    var = np.var(x)
    if var < 1e-15:
        return np.zeros(max_lag + 1)
    acf = np.zeros(max_lag + 1)
    n = len(x)
    for k in range(max_lag + 1):
        acf[k] = np.sum(x_centered[:n - k] * x_centered[k:]) / (n * var)
    return acf


def spectral_density_ar1(rho, sigma, freqs):
    """Analytical spectral density of AR(1): sigma^2 / |1 - rho*e^{-i*omega}|^2."""
    denom = np.abs(1 - rho * np.exp(-1j * freqs)) ** 2
    return sigma**2 / (2 * np.pi * denom)


def main():
    # =========================================================================
    # Parse the Dynare .mod files
    # =========================================================================
    dynare_dir = Path(__file__).resolve().parents[1]
    ar1_mod = parse_mod_file(dynare_dir / "ar1" / "model.mod")
    ar2_mod = parse_mod_file(dynare_dir / "ar2" / "model.mod")
    print("Parsed ar1/model.mod and ar2/model.mod")

    # =========================================================================
    # AR(1) Parameters (from ar1/model.mod)
    # =========================================================================
    rho_ar1 = 0.9
    sigma_ar1 = 0.01

    # Multiplier-Accelerator Parameters (from ar2/model.mod)
    alpha_ma = 0.3   # accelerator coefficient
    beta_ma = 0.8    # marginal propensity to consume
    rho_ma = 0.9     # government spending persistence
    sigma_ma = 0.01
    Gbar = 1.0

    # =========================================================================
    # Simulations
    # =========================================================================
    T_sim = 200
    T_irf = 40
    print("Simulating AR(1) and multiplier-accelerator processes...")

    # AR(1) simulation
    ar1_path, ar1_shocks = simulate_ar1(rho_ar1, sigma_ar1, T_sim)

    # Multiplier-accelerator simulation
    ma_sim = simulate_multiplier_accelerator(alpha_ma, beta_ma, rho_ma, sigma_ma, Gbar, T_sim)

    # IRFs
    ar1_irf = irf_ar1(rho_ar1, T_irf)

    # Also compute IRFs for different rho values for comparison
    rho_values = [0.5, 0.7, 0.9, 0.99]
    ar1_irfs = {rho: irf_ar1(rho, T_irf) for rho in rho_values}

    ma_irf = irf_multiplier_accelerator(alpha_ma, beta_ma, rho_ma, Gbar, T_irf)

    # Autocorrelations
    ar1_acf = autocorrelation(ar1_path, max_lag=20)
    ma_y_acf = autocorrelation(ma_sim["Y"], max_lag=20)

    # Spectral density
    freqs = np.linspace(0.01, np.pi, 500)

    print("  Simulations and analysis complete.")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "AR Process Dynamics",
        "Impulse responses, simulated paths, and spectral properties of AR(1) and "
        "multiplier-accelerator models from Dynare.",
    )

    report.add_overview(
        "Autoregressive processes are the building blocks of time series econometrics "
        "and DSGE modeling. Every linearized DSGE model reduces to a vector autoregression "
        "in the state variables.\n\n"
        "This module analyzes two models from the Dynare examples:\n"
        "1. **AR(1):** The simplest persistent process, $x_t = \\rho x_{t-1} + \\varepsilon_t$\n"
        "2. **Multiplier-Accelerator:** A classic Keynesian dynamics model where consumption "
        "depends on lagged income (multiplier) and investment depends on consumption changes "
        "(accelerator), producing richer dynamics than a simple AR."
    )

    report.add_equations(
        r"""
**AR(1) Process** (`ar1/model.mod`):
```
x = rho*x(-1) + e
```
$$x_t = \rho \, x_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim N(0, \sigma^2)$$

**Multiplier-Accelerator Model** (`ar2/model.mod`):
```
C = beta * Y(-1)           [Consumption function]
G = rho * G(-1) + (1-rho)*Gbar + e   [Government spending]
I = alpha * (C - C(-1))    [Accelerator investment]
Y = C + I + G              [Income identity]
```
$$C_t = \beta Y_{t-1}$$
$$G_t = \rho G_{t-1} + (1-\rho)\bar{G} + \varepsilon_t$$
$$I_t = \alpha (C_t - C_{t-1})$$
$$Y_t = C_t + I_t + G_t$$
"""
    )

    report.add_model_setup(
        "**AR(1) Parameters:**\n\n"
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\rho$   | {rho_ar1} | Persistence |\n"
        f"| $\\sigma$ | {sigma_ar1} | Shock std. dev. |\n\n"
        "**Multiplier-Accelerator Parameters:**\n\n"
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\alpha$  | {alpha_ma} | Accelerator coefficient |\n"
        f"| $\\beta$   | {beta_ma} | Marginal propensity to consume |\n"
        f"| $\\rho$    | {rho_ma} | Government spending persistence |\n"
        f"| $\\bar{{G}}$ | {Gbar} | Steady-state government spending |\n"
        f"| $\\sigma$  | {sigma_ma} | Shock std. dev. |"
    )

    report.add_solution_method(
        "These models are **purely backward-looking**, so no expectations need to be "
        "solved --- the system can be simulated forward directly.\n\n"
        "**AR(1) properties (analytical):**\n"
        f"- Unconditional mean: $E[x] = 0$\n"
        f"- Unconditional variance: $\\sigma^2_x = \\sigma^2 / (1-\\rho^2) = "
        f"{sigma_ar1**2 / (1 - rho_ar1**2):.6f}$\n"
        f"- Autocorrelation at lag $k$: $\\rho^k$\n"
        f"- Half-life: $\\ln(0.5)/\\ln(\\rho) = {np.log(0.5)/np.log(rho_ar1):.1f}$ periods\n\n"
        "**Multiplier-Accelerator:** The interaction of the consumption multiplier "
        "($\\beta$) and investment accelerator ($\\alpha$) can produce oscillatory dynamics "
        "when the characteristic roots are complex."
    )

    periods = np.arange(T_irf)

    # --- Figure 1: AR(1) IRFs for different persistence levels ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = ["#abd9e9", "#74add1", "#2c7bb6", "#053061"]
    for (rho_val, irf), color in zip(ar1_irfs.items(), colors):
        ax1.plot(periods, irf, color=color, linewidth=2.5, label=f"$\\rho = {rho_val}$")
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax1.set_xlabel("Periods after shock")
    ax1.set_ylabel("Response to unit shock")
    ax1.set_title("AR(1) Impulse Response Functions")
    ax1.legend(fontsize=11)
    report.add_figure(
        "figures/ar1-irfs.png",
        "AR(1) impulse responses for different persistence parameters",
        fig1,
        description="Higher persistence (darker blue) means shocks decay more slowly and have longer-lasting "
        "effects. At rho=0.99 the process is near a unit root and shocks are almost permanent, while "
        "at rho=0.5 the half-life is just one period. This tradeoff between persistence and mean-reversion "
        "is central to calibrating TFP processes in DSGE models.",
    )

    # --- Figure 2: Multiplier-Accelerator IRFs ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))
    ma_vars = [("Y", "Output ($Y$)", "#2c7bb6"),
               ("C", "Consumption ($C$)", "#d7191c"),
               ("I", "Investment ($I$)", "#fdae61"),
               ("G", "Government ($G$)", "#018571")]
    for ax, (key, title, color) in zip(axes2.flatten(), ma_vars):
        ax.plot(periods, ma_irf[key], color=color, linewidth=2.5)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Periods after shock")
        ax.set_ylabel("Response")
        ax.set_title(title)
    fig2.suptitle("Multiplier-Accelerator: IRFs to Unit Government Spending Shock",
                  fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    report.add_figure(
        "figures/multiplier-accelerator-irfs.png",
        "Multiplier-accelerator impulse responses to a unit government spending shock",
        fig2,
        description="The output response exceeds the initial government spending impulse because the "
        "consumption multiplier (beta) feeds back into income, and the investment accelerator (alpha) "
        "amplifies changes in consumption. Notice how investment can overshoot and oscillate, a signature "
        "of the accelerator mechanism responding to the derivative of consumption.",
    )

    # --- Figure 3: Simulated paths ---
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 8))

    t_plot = np.arange(T_sim)
    ax3a.plot(t_plot, ar1_path, "#2c7bb6", linewidth=1)
    ax3a.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax3a.set_xlabel("Period")
    ax3a.set_ylabel("$x_t$")
    ax3a.set_title(f"Simulated AR(1) Path ($\\rho={rho_ar1}$, $\\sigma={sigma_ar1}$)")

    ax3b.plot(t_plot, ma_sim["Y"], "#2c7bb6", linewidth=1, label="$Y$")
    ax3b.plot(t_plot, ma_sim["C"], "#d7191c", linewidth=1, label="$C$", alpha=0.8)
    ax3b.plot(t_plot, ma_sim["G"], "#018571", linewidth=1, label="$G$", alpha=0.8)
    ax3b.set_xlabel("Period")
    ax3b.set_ylabel("Level")
    ax3b.set_title("Simulated Multiplier-Accelerator Paths")
    ax3b.legend()

    fig3.tight_layout()
    report.add_figure(
        "figures/simulated-paths.png",
        "Simulated time series for AR(1) and multiplier-accelerator models",
        fig3,
        description="The AR(1) path (top) shows the characteristic smooth, mean-reverting fluctuations "
        "of a highly persistent process. The multiplier-accelerator paths (bottom) display richer dynamics: "
        "consumption closely tracks output with a lag, while government spending is the exogenous driver.",
    )

    # --- Figure 4: Autocorrelation functions ---
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))

    lags = np.arange(len(ar1_acf))
    # Theoretical AR(1) ACF
    ar1_acf_theoretical = rho_ar1 ** lags

    ax4a.bar(lags, ar1_acf, color="#2c7bb6", alpha=0.6, label="Sample ACF")
    ax4a.plot(lags, ar1_acf_theoretical, "r-o", markersize=4, linewidth=1.5,
              label=f"Theoretical $\\rho^k$")
    ax4a.axhline(0, color="black", linewidth=0.5)
    ax4a.set_xlabel("Lag $k$")
    ax4a.set_ylabel("Autocorrelation")
    ax4a.set_title("AR(1) Autocorrelation Function")
    ax4a.legend()

    ax4b.bar(lags, ma_y_acf, color="#d7191c", alpha=0.6)
    ax4b.axhline(0, color="black", linewidth=0.5)
    ax4b.set_xlabel("Lag $k$")
    ax4b.set_ylabel("Autocorrelation")
    ax4b.set_title("Multiplier-Accelerator: Output ACF")

    fig4.tight_layout()
    report.add_figure(
        "figures/autocorrelation.png",
        "Autocorrelation functions: AR(1) sample vs theoretical, and multiplier-accelerator output",
        fig4,
        description="The AR(1) sample ACF (left) closely matches the theoretical rho^k decay, validating "
        "the simulation. The multiplier-accelerator ACF (right) can exhibit a non-monotone pattern due to "
        "the interaction of the multiplier and accelerator channels, which can produce damped oscillations "
        "in the autocorrelation structure.",
    )

    # --- Figure 5: Spectral density ---
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    for rho_val, color in zip([0.5, 0.9, 0.99], ["#abd9e9", "#2c7bb6", "#053061"]):
        sd = spectral_density_ar1(rho_val, sigma_ar1, freqs)
        ax5.plot(freqs, sd, color=color, linewidth=2, label=f"$\\rho = {rho_val}$")
    ax5.set_xlabel("Frequency $\\omega$")
    ax5.set_ylabel("Spectral density $S(\\omega)$")
    ax5.set_title("Spectral Density of AR(1) Processes")
    ax5.legend()
    ax5.set_yscale("log")
    report.add_figure(
        "figures/spectral-density.png",
        "Spectral density of AR(1) for different persistence levels (log scale)",
        fig5,
        description="Higher persistence concentrates spectral power at low frequencies (long cycles) and "
        "suppresses high-frequency variation. At rho=0.99 the spectrum is almost flat at high frequencies "
        "but explodes near zero, showing that near-unit-root processes look like slow-moving trends. "
        "This frequency-domain view explains why persistent TFP shocks generate smooth business cycles.",
    )

    # --- Table ---
    ar_summary = {
        "Property": [
            "Persistence ($\\rho$)",
            "Unconditional variance",
            "Half-life (periods)",
            "Spectral peak",
        ],
        "AR(1), $\\rho=0.5$": [
            "0.5",
            f"{sigma_ar1**2/(1-0.5**2):.6f}",
            f"{np.log(0.5)/np.log(0.5):.1f}",
            "Frequency 0 (low-pass)",
        ],
        "AR(1), $\\rho=0.9$": [
            "0.9",
            f"{sigma_ar1**2/(1-0.9**2):.6f}",
            f"{np.log(0.5)/np.log(0.9):.1f}",
            "Frequency 0 (low-pass)",
        ],
        "AR(1), $\\rho=0.99$": [
            "0.99",
            f"{sigma_ar1**2/(1-0.99**2):.6f}",
            f"{np.log(0.5)/np.log(0.99):.1f}",
            "Frequency 0 (low-pass)",
        ],
    }
    df = pd.DataFrame(ar_summary)
    report.add_table("tables/ar-properties.csv", "AR(1) Process Properties", df,
        description="The unconditional variance rises sharply with persistence because shocks accumulate "
        "rather than dissipating. The half-life jumps from 1 period at rho=0.5 to 69 periods at rho=0.99, "
        "illustrating why the choice of persistence parameter is so consequential for DSGE calibration.")

    report.add_takeaway(
        "Autoregressive dynamics are the foundation of time series econometrics and "
        "macroeconomic modeling.\n\n"
        "**Key insights:**\n"
        "- AR(1) persistence ($\\rho$) controls the half-life of shocks: at $\\rho=0.9$, "
        f"a shock takes {np.log(0.5)/np.log(0.9):.0f} periods to decay by half. At "
        f"$\\rho=0.99$, this rises to {np.log(0.5)/np.log(0.99):.0f} periods.\n"
        "- Higher persistence concentrates spectral power at low frequencies, meaning "
        "the process exhibits long, smooth cycles rather than rapid fluctuations.\n"
        "- The multiplier-accelerator model shows how interaction between consumption "
        "(multiplier: $\\beta$) and investment (accelerator: $\\alpha$) can produce "
        "oscillatory dynamics even from monotone AR(1) government spending.\n"
        "- The accelerator effect amplifies shocks: investment responds to *changes* in "
        "consumption, creating a derivative-like feedback that can overshoot.\n"
        "- Understanding AR dynamics is essential because every linearized DSGE model "
        "reduces to a VAR in its state variables."
    )

    report.add_references([
        "Hamilton, J. (1994). *Time Series Analysis*. Princeton University Press.",
        "Samuelson, P. (1939). Interactions between the Multiplier Analysis and the Principle of Acceleration. *Review of Economics and Statistics*, 21(2), 75-78.",
        "Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 2.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
