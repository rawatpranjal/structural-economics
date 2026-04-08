#!/usr/bin/env python3
"""Diamond-Mortensen-Pissarides (DMP) Search and Matching Model.

Simulates the canonical DMP model of equilibrium unemployment with aggregate
productivity shocks. Computes the linearized equilibrium dynamics of labor
market tightness, unemployment, and vacancies.

Reference: Shimer (2005), "The Cyclical Behavior of Equilibrium Unemployment
and Vacancies," American Economic Review.
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def main():
    # =========================================================================
    # Parameters (monthly calibration, Shimer 2005)
    # =========================================================================
    beta = 0.996       # Monthly discount factor
    rho = 0.949        # Persistence of productivity shock (AR(1))
    sigma_e = 0.0065   # Std dev of productivity innovation
    sigma = 0.034      # Separation rate (exogenous)
    chi = 0.49         # Matching function elasticity (m = chi * v^eta * u^(1-eta))
    b = 0.4            # Unemployment benefit (flow value of leisure)
    gamma = 0.72       # Worker's bargaining power
    eta = 0.72         # Matching function elasticity wrt vacancies
    z_bar = 1.0        # Steady-state productivity

    # =========================================================================
    # Steady-state computation
    # =========================================================================
    # Linearized equilibrium coefficients
    phi1 = (1 - sigma - gamma * chi) / (1 - gamma) / chi
    phi2 = (1 - gamma) * beta * chi
    k = (z_bar - b) * phi2 / (1 - phi1 * phi2)  # Vacancy posting cost
    A_coeff = eta * k / ((1 - gamma) * beta * chi)
    B_coeff = beta * A_coeff * (1 - sigma) - gamma * k / (1 - gamma)
    C_coeff = rho / (A_coeff - B_coeff * rho)  # Elasticity of theta wrt z

    # Steady-state values
    theta_ss = 1.0   # Normalized
    q_ss = chi * theta_ss ** (eta - 1)  # Probability of filling a vacancy
    f_ss = chi * theta_ss ** eta        # Job finding rate
    u_ss = sigma / (sigma + f_ss)       # Steady-state unemployment rate
    w_ss = gamma * (z_bar + k * theta_ss) + (1 - gamma) * b  # Nash wage

    print(f"Steady state: u={u_ss:.4f}, theta={theta_ss:.4f}, w={w_ss:.4f}")
    print(f"Elasticity C (d_log_theta / d_log_z) = {C_coeff:.4f}")

    # =========================================================================
    # Simulation
    # =========================================================================
    np.random.seed(42)
    T = 5000
    burn = 500

    # Productivity shock (log-linearized)
    zhat = np.zeros(T)
    for t in range(1, T):
        zhat[t] = rho * zhat[t - 1] + np.random.normal(0, sigma_e)

    z = z_bar * np.exp(zhat)

    # Labor market tightness (log-linearized)
    theta_hat = C_coeff * zhat
    theta = np.exp(theta_hat)

    # Job finding rate and unemployment dynamics
    f = chi * theta ** eta
    u = np.zeros(T)
    u[0] = u_ss
    for t in range(1, T):
        u[t] = sigma * (1 - u[t - 1]) + (1 - f[t - 1]) * u[t - 1]

    v = theta * u  # Vacancies

    # Trim burn-in
    z_plot = z[burn:]
    u_plot = u[burn:]
    v_plot = v[burn:]
    theta_plot = theta[burn:]
    T_plot = len(z_plot)

    # =========================================================================
    # Compute statistics
    # =========================================================================
    from scipy.signal import detrend

    stats = {
        "Variable": ["Productivity z", "Unemployment u", "Vacancies v", "Tightness theta", "v/u ratio"],
        "Mean": [f"{np.mean(z_plot):.4f}", f"{np.mean(u_plot):.4f}", f"{np.mean(v_plot):.4f}",
                 f"{np.mean(theta_plot):.4f}", f"{np.mean(v_plot / u_plot):.4f}"],
        "Std Dev": [f"{np.std(z_plot):.4f}", f"{np.std(u_plot):.4f}", f"{np.std(v_plot):.4f}",
                    f"{np.std(theta_plot):.4f}", f"{np.std(v_plot / u_plot):.4f}"],
        "Corr(x, z)": [f"{np.corrcoef(z_plot, z_plot)[0, 1]:.4f}",
                        f"{np.corrcoef(u_plot, z_plot)[0, 1]:.4f}",
                        f"{np.corrcoef(v_plot, z_plot)[0, 1]:.4f}",
                        f"{np.corrcoef(theta_plot, z_plot)[0, 1]:.4f}",
                        f"{np.corrcoef(v_plot / u_plot, z_plot)[0, 1]:.4f}"],
    }

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Diamond-Mortensen-Pissarides Model",
        "Equilibrium search and matching model of unemployment with aggregate productivity shocks.",
    )

    report.add_overview(
        "The DMP model is the workhorse framework for analyzing labor market dynamics. "
        "Firms post vacancies at a cost, unemployed workers search for jobs, and wages are "
        "determined by Nash bargaining. The key state variable is *labor market tightness* "
        "$\\theta = v/u$ (vacancies per unemployed worker), which determines both the job "
        "finding rate and the vacancy filling rate through a matching function.\n\n"
        "This implementation follows Shimer (2005), who showed that the standard DMP model "
        "generates too little unemployment volatility relative to the data — the *Shimer puzzle*."
    )

    report.add_equations(r"""
**Matching function:** $m = \chi \cdot v^{\eta} \cdot u^{1-\eta}$

**Job finding rate:** $f(\theta) = \chi \cdot \theta^{\eta}$, where $\theta = v/u$

**Vacancy filling rate:** $q(\theta) = \chi \cdot \theta^{\eta - 1}$

**Free entry (vacancy creation):**
$$\frac{k}{q(\theta)} = \beta \left[ (1-\gamma)(z - b) + \frac{k \cdot \theta}{1-\gamma} \cdot \gamma + (1-\sigma) \frac{k}{q(\theta)} \right]$$

**Nash bargaining wage:**
$$w = \gamma (z + k\theta) + (1-\gamma) b$$

**Unemployment dynamics:**
$$u_{t+1} = \sigma (1 - u_t) + (1 - f(\theta_t)) u_t$$

**Log-linearized tightness response:** $\hat{\theta}_t = C \cdot \hat{z}_t$, where $C = \frac{\rho}{A - B\rho}$
""")

    report.add_model_setup(
        "| Parameter | Value | Description |\n"
        "|-----------|-------|-------------|\n"
        f"| $\\beta$ | {beta} | Monthly discount factor |\n"
        f"| $\\rho$ | {rho} | Productivity persistence |\n"
        f"| $\\sigma_e$ | {sigma_e} | Productivity innovation std |\n"
        f"| $\\sigma$ | {sigma} | Separation rate |\n"
        f"| $\\chi$ | {chi} | Matching efficiency |\n"
        f"| $b$ | {b} | Unemployment benefit |\n"
        f"| $\\gamma$ | {gamma} | Worker bargaining power |\n"
        f"| $\\eta$ | {eta} | Matching elasticity |"
    )

    report.add_solution_method(
        "The model is solved by log-linearization around the steady state. "
        "The key object is the elasticity $C$ of labor market tightness with respect to "
        f"productivity: $C = {C_coeff:.4f}$. This means a 1% increase in productivity leads to a "
        f"{C_coeff:.2f}% increase in tightness.\n\n"
        "Simulation: 5,000 periods of aggregate productivity shocks drawn from "
        f"$\\hat{{z}}_{{t+1}} = {rho} \\hat{{z}}_t + \\epsilon_t$, $\\epsilon_t \\sim N(0, {sigma_e}^2)$. "
        "Unemployment evolves according to the flow equation."
    )

    # --- Figure 1: Unemployment and Vacancies ---
    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    t_axis = np.arange(T_plot)
    ax1a.plot(t_axis[:1000], u_plot[:1000], "b-", linewidth=1, label="Unemployment $u$")
    ax1a.set_ylabel("Unemployment rate")
    ax1a.set_title("Unemployment Dynamics")
    ax1a.legend()

    ax1b.plot(t_axis[:1000], v_plot[:1000], "r-", linewidth=1, label="Vacancies $v$")
    ax1b.set_ylabel("Vacancy rate")
    ax1b.set_xlabel("Period")
    ax1b.set_title("Vacancy Dynamics")
    ax1b.legend()
    fig1.tight_layout()
    report.add_figure("figures/unemployment-vacancies.png", "Unemployment and vacancy dynamics over 1000 periods", fig1,
        description="Unemployment and vacancies move in opposite directions: when firms post more vacancies in booms, "
        "the job finding rate rises and unemployment falls. The relative amplitudes reveal how much the labor market amplifies small productivity shocks.")

    # --- Figure 2: Beveridge Curve ---
    fig2, ax2 = plt.subplots()
    ax2.scatter(u_plot, v_plot, s=1, alpha=0.3, c="steelblue")
    ax2.set_xlabel("Unemployment rate $u$")
    ax2.set_ylabel("Vacancy rate $v$")
    ax2.set_title("Beveridge Curve")
    report.add_figure("figures/beveridge-curve.png", "Beveridge curve: negative correlation between unemployment and vacancies", fig2,
        description="The downward-sloping scatter is the Beveridge curve, one of the most robust empirical regularities in labor economics. "
        "The tightness of the cloud reflects the strength of the matching function; shifts of the entire curve would indicate changes in matching efficiency.")

    # --- Figure 3: Productivity and Tightness ---
    fig3, ax3 = plt.subplots()
    ax3.plot(t_axis[:500], z_plot[:500], "k-", linewidth=1, label="Productivity $z$")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(t_axis[:500], theta_plot[:500], "r-", linewidth=1, label="Tightness $\\theta$", alpha=0.7)
    ax3.set_xlabel("Period")
    ax3.set_ylabel("Productivity $z$")
    ax3_twin.set_ylabel("Market tightness $\\theta$")
    ax3.set_title("Productivity and Labor Market Tightness")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    report.add_figure("figures/productivity-tightness.png", "Productivity shocks drive labor market tightness (amplification factor C)", fig3,
        description="Tightness responds proportionally to productivity via the elasticity C, but with the standard calibration (b=0.4) "
        "the amplification is modest. This is the Shimer puzzle: the model needs a much higher b (closer to z) to match the observed volatility of labor market tightness.")

    # --- Table ---
    df = pd.DataFrame(stats)
    report.add_table("tables/business-cycle-stats.csv", "Business Cycle Statistics (simulated)", df,
        description="The key diagnostic is the standard deviation of unemployment relative to productivity: "
        "in U.S. data this ratio is roughly 20, but the baseline DMP calibration generates far less amplification.")

    report.add_takeaway(
        "The DMP model captures the key qualitative features of labor market dynamics:\n\n"
        "**Key insights:**\n"
        "- The **Beveridge curve**: unemployment and vacancies are negatively correlated. "
        "In booms, firms post more vacancies and unemployment falls; in recessions, the reverse.\n"
        "- **Amplification**: small productivity shocks generate fluctuations in tightness, "
        f"unemployment, and vacancies. The elasticity $C = {C_coeff:.2f}$ measures this amplification.\n"
        "- **The Shimer puzzle**: with standard calibration ($b = 0.4$, $\\gamma = 0.72$), "
        "the model generates too little unemployment volatility compared to U.S. data. "
        "The unemployment rate barely moves in response to productivity shocks.\n"
        "- **Resolution**: Hagedorn and Manovskii (2008) show that setting $b$ close to productivity "
        "(e.g., $b = 0.95$) dramatically increases amplification, as small surplus changes cause "
        "large vacancy responses."
    )

    report.add_references([
        "Diamond, P. (1982). \"Aggregate Demand Management in Search Equilibrium.\" *Journal of Political Economy*, 90(5).",
        "Mortensen, D. and Pissarides, C. (1994). \"Job Creation and Job Destruction in the Theory of Unemployment.\" *Review of Economic Studies*, 61(3).",
        "Shimer, R. (2005). \"The Cyclical Behavior of Equilibrium Unemployment and Vacancies.\" *American Economic Review*, 95(1).",
        "Hagedorn, M. and Manovskii, I. (2008). \"The Cyclical Behavior of Equilibrium Unemployment and Vacancies Revisited.\" *American Economic Review*, 98(4).",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
