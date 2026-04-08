#!/usr/bin/env python3
"""RBC with Capital Taxation: How Tax Distortions Affect Investment.

Solves an RBC model with a time-varying capital income tax tau_k using global
VFI. The government collects tau_k * r * K and returns it lump-sum. Compares
optimal policy under different tax rates to quantify how capital taxation
distorts the steady-state capital stock and investment dynamics.

Reference: Cole and Obstfeld (1991), GDSGE toolbox (Cao, Luo, Nie 2023).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def solve_rbc_tax(tau_k, beta=0.99, alpha=0.36, sigma=2.0, delta=0.025,
                  rho=0.95, sigma_e=0.01, n_k=40, n_z=5, tol=1e-6, max_iter=500,
                  verbose=True):
    """Solve the RBC model with capital tax tau_k via VFI.

    Returns policy functions, value function, grids, and steady-state values.
    """
    # Steady state with tax: (1-tau_k) * alpha * K^(alpha-1) = 1/beta - 1 + delta
    Kss = ((1.0 - tau_k) * alpha / (1.0 / beta - 1.0 + delta)) ** (1.0 / (1.0 - alpha))
    Yss = Kss ** alpha
    Css = Yss - delta * Kss
    Iss = delta * Kss

    # Capital grid
    K_min = max(Kss * 0.7, 0.1)
    K_max = Kss * 1.3
    K_grid = np.linspace(K_min, K_max, n_k)

    # TFP grid (Tauchen)
    from scipy.stats import norm
    sigma_z = sigma_e / np.sqrt(1.0 - rho ** 2)
    m_z = 3.0
    z_grid_log = np.linspace(-m_z * sigma_z, m_z * sigma_z, n_z)
    z_grid = np.exp(z_grid_log)
    step = z_grid_log[1] - z_grid_log[0]

    trans_z = np.zeros((n_z, n_z))
    for i in range(n_z):
        for j in range(n_z):
            if j == 0:
                trans_z[i, j] = norm.cdf((z_grid_log[j] - rho * z_grid_log[i] + step / 2) / sigma_e)
            elif j == n_z - 1:
                trans_z[i, j] = 1.0 - norm.cdf((z_grid_log[j] - rho * z_grid_log[i] - step / 2) / sigma_e)
            else:
                trans_z[i, j] = (
                    norm.cdf((z_grid_log[j] - rho * z_grid_log[i] + step / 2) / sigma_e)
                    - norm.cdf((z_grid_log[j] - rho * z_grid_log[i] - step / 2) / sigma_e)
                )

    def u(c):
        return np.where(c > 1e-10, c ** (1.0 - sigma) / (1.0 - sigma), -1e10)

    # Initialize V
    V = np.zeros((n_z, n_k))
    for iz in range(n_z):
        for ik in range(n_k):
            c_guess = max(z_grid[iz] * K_grid[ik] ** alpha + (1.0 - delta) * K_grid[ik] - Kss, 1e-10)
            V[iz, ik] = u(np.array([c_guess]))[0] / (1.0 - beta)

    policy_k = np.zeros((n_z, n_k))
    policy_c = np.zeros((n_z, n_k))
    policy_idx = np.zeros((n_z, n_k), dtype=int)
    howard_steps = 25

    # Precompute utility matrices
    u_mats = np.zeros((n_z, n_k, n_k))
    resources_all = np.zeros((n_z, n_k))
    for iz in range(n_z):
        resources_all[iz] = z_grid[iz] * K_grid ** alpha + (1.0 - delta) * K_grid
        c_mat = resources_all[iz][:, None] - K_grid[None, :]
        u_mats[iz] = u(c_mat)

    for iteration in range(1, max_iter + 1):
        V_new = np.zeros_like(V)

        for iz in range(n_z):
            EV_kprime = trans_z[iz, :] @ V
            val_mat = u_mats[iz] + beta * EV_kprime[None, :]
            best_idx = np.argmax(val_mat, axis=1)
            V_new[iz, :] = val_mat[np.arange(n_k), best_idx]
            policy_idx[iz, :] = best_idx
            policy_k[iz, :] = K_grid[best_idx]
            policy_c[iz, :] = resources_all[iz] - K_grid[best_idx]

        error = np.max(np.abs(V_new - V))
        V = V_new.copy()

        # Howard policy iteration acceleration
        for _ in range(howard_steps):
            V_howard = np.zeros_like(V)
            for iz in range(n_z):
                EV_kprime = trans_z[iz, :] @ V
                for ik in range(n_k):
                    ik_prime = policy_idx[iz, ik]
                    V_howard[iz, ik] = u_mats[iz][ik, ik_prime] + beta * EV_kprime[ik_prime]
            V = V_howard

        if verbose and iteration % 10 == 0:
            print(f"    tau={tau_k:.2f} VFI iter {iteration:3d}, error = {error:.2e}")
        if error < tol:
            if verbose:
                print(f"    tau={tau_k:.2f} converged in {iteration} iters (error = {error:.2e})")
            break

    # Euler-based refinement: incorporate the tax wedge into the consumption
    # policy via iteration on the after-tax Euler equation.
    for euler_iter in range(300):
        policy_c_new = np.zeros_like(policy_c)
        for iz in range(n_z):
            resources = resources_all[iz]
            for ik in range(n_k):
                kp = policy_k[iz, ik]
                Ec = 0.0
                for jz in range(n_z):
                    z_next = z_grid[jz]
                    c_next = np.interp(kp, K_grid, policy_c[jz, :])
                    mpk_next = (1.0 - tau_k) * alpha * z_next * kp ** (alpha - 1.0) + 1.0 - delta
                    Ec += trans_z[iz, jz] * c_next ** (-sigma) * mpk_next
                c_euler = (beta * Ec) ** (-1.0 / sigma)
                c_euler = np.clip(c_euler, 1e-10, resources[ik] - K_min)
                policy_c_new[iz, ik] = c_euler
            policy_k[iz, :] = np.clip(resources - policy_c_new[iz, :], K_min, K_max)

        err_euler = np.max(np.abs(policy_c_new - policy_c))
        policy_c = policy_c_new.copy()
        if err_euler < tol:
            if verbose:
                print(f"    tau={tau_k:.2f} Euler refinement converged in {euler_iter+1} iters")
            break

    return {
        "V": V, "policy_k": policy_k, "policy_c": policy_c,
        "K_grid": K_grid, "z_grid": z_grid, "trans_z": trans_z,
        "Kss": Kss, "Yss": Yss, "Css": Css, "Iss": Iss,
        "K_min": K_min, "K_max": K_max,
        "tau_k": tau_k, "iterations": iteration,
    }


def simulate(sol, T=5000, seed=42):
    """Simulate the model for T periods."""
    np.random.seed(seed)
    n_z = len(sol["z_grid"])
    K_grid = sol["K_grid"]
    z_grid = sol["z_grid"]
    trans_z = sol["trans_z"]
    alpha = 0.36
    delta = 0.025

    interp_k = RegularGridInterpolator(
        (z_grid, K_grid), sol["policy_k"], method="linear",
        bounds_error=False, fill_value=None
    )
    interp_c = RegularGridInterpolator(
        (z_grid, K_grid), sol["policy_c"], method="linear",
        bounds_error=False, fill_value=None
    )

    z_idx = np.zeros(T, dtype=int)
    z_idx[0] = n_z // 2
    for t in range(T - 1):
        z_idx[t + 1] = min(
            np.searchsorted(np.cumsum(trans_z[z_idx[t], :]), np.random.uniform()),
            n_z - 1
        )
    z_sim = z_grid[z_idx]

    K_sim = np.zeros(T)
    C_sim = np.zeros(T)
    Y_sim = np.zeros(T)
    I_sim = np.zeros(T)
    K_sim[0] = sol["Kss"]

    for t in range(T):
        pt = np.array([[z_sim[t], K_sim[t]]])
        C_sim[t] = float(interp_c(pt))
        Y_sim[t] = z_sim[t] * K_sim[t] ** alpha
        kp = float(interp_k(pt))
        I_sim[t] = kp - (1.0 - delta) * K_sim[t]
        if t < T - 1:
            K_sim[t + 1] = np.clip(kp, sol["K_min"], sol["K_max"])

    return {"K": K_sim, "C": C_sim, "Y": Y_sim, "I": I_sim, "z": z_sim}


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    beta = 0.99
    alpha = 0.36
    sigma = 2.0
    delta = 0.025
    rho = 0.95
    sigma_e = 0.01

    tau_values = [0.0, 0.10, 0.20, 0.30, 0.40]

    # =========================================================================
    # Solve for each tax rate
    # =========================================================================
    print("Solving RBC-Capital-Tax model for multiple tax rates...")
    solutions = {}
    for tau_k in tau_values:
        print(f"\n  Solving tau_k = {tau_k:.2f}...")
        solutions[tau_k] = solve_rbc_tax(
            tau_k, beta=beta, alpha=alpha, sigma=sigma, delta=delta,
            rho=rho, sigma_e=sigma_e, n_k=40, n_z=5, verbose=True
        )

    # =========================================================================
    # Simulate each
    # =========================================================================
    T_sim = 5000
    burn = 500
    simulations = {}
    for tau_k in tau_values:
        simulations[tau_k] = simulate(solutions[tau_k], T=T_sim, seed=42)

    # =========================================================================
    # Steady state analysis
    # =========================================================================
    ss_data = []
    Kss_notax = solutions[0.0]["Kss"]
    Yss_notax = solutions[0.0]["Yss"]
    Css_notax = solutions[0.0]["Css"]
    for tau_k in tau_values:
        sol = solutions[tau_k]
        sim = simulations[tau_k]
        ss_data.append({
            "Tax rate": f"{tau_k:.0%}",
            "K_ss": f"{sol['Kss']:.4f}",
            "Y_ss": f"{sol['Yss']:.4f}",
            "C_ss": f"{sol['Css']:.4f}",
            "K_ss / K_ss(0)": f"{sol['Kss']/Kss_notax:.3f}",
            "Mean K (sim)": f"{sim['K'][burn:].mean():.4f}",
            "std(Y) %": f"{100*np.std(np.log(sim['Y'][burn:]) - np.log(sim['Y'][burn:]).mean()):.3f}",
        })

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "RBC with Capital Taxation",
        "How capital income taxes distort investment, steady-state capital, and business cycle dynamics.",
    )

    report.add_overview(
        "This model extends the standard RBC framework with a capital income tax $\\tau_k$. "
        "The government taxes the return on capital at rate $\\tau_k$ and returns the revenue "
        "as a lump-sum transfer. Even though the transfer makes the tax revenue-neutral, "
        "the tax wedge distorts the intertemporal margin: the after-tax return on saving is "
        "$(1-\\tau_k) r$, reducing the incentive to accumulate capital.\n\n"
        "We solve globally for five tax rates (0% to 40%) and compare the resulting "
        "steady states, policy functions, and business cycle dynamics."
    )

    report.add_equations(
        r"""
$$V(K, z) = \max_{c, K'} \left\{ u(c) + \beta \, \mathbb{E}\left[V(K', z')\right] \right\}$$

**Budget constraint (with lump-sum rebate):**
$$c + K' = z K^\alpha + (1-\delta) K$$

**Euler equation (after-tax):**
$$c^{-\sigma} = \beta \, \mathbb{E}\left[ c'^{-\sigma} \left((1-\tau_k) \alpha z' K'^{\alpha-1} + 1 - \delta\right) \right]$$

**Steady state capital:**
$$K_{ss}(\tau_k) = \left(\frac{(1-\tau_k)\alpha}{1/\beta - 1 + \delta}\right)^{\frac{1}{1-\alpha}}$$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\beta$  | {beta} | Discount factor |\n"
        f"| $\\alpha$ | {alpha} | Capital share |\n"
        f"| $\\sigma$ | {sigma} | CRRA coefficient |\n"
        f"| $\\delta$ | {delta} | Depreciation rate |\n"
        f"| $\\rho$   | {rho} | TFP persistence |\n"
        f"| $\\sigma_\\varepsilon$ | {sigma_e} | TFP innovation std |\n"
        f"| $\\tau_k$ | {tau_values} | Tax rates compared |"
    )

    report.add_solution_method(
        "**Value Function Iteration** followed by **Euler equation refinement**. "
        "For each tax rate, we first solve the planner's VFI on a 40x5 grid, then "
        "refine the consumption policy using the after-tax Euler equation with the "
        "correct tax wedge on the marginal product of capital.\n\n"
        "The tax does not change the budget set (due to lump-sum rebate) but alters "
        "the first-order condition, driving a wedge between the marginal rate of "
        "substitution and the marginal rate of transformation."
    )

    # --- Figure 1: Steady state capital vs tax rate ---
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(13, 5))
    tau_plot = np.linspace(0, 0.5, 100)
    Kss_plot = ((1.0 - tau_plot) * alpha / (1.0 / beta - 1.0 + delta)) ** (1.0 / (1.0 - alpha))
    Yss_plot = Kss_plot ** alpha
    Css_plot = Yss_plot - delta * Kss_plot

    ax1a.plot(tau_plot * 100, Kss_plot, "b-", linewidth=2, label="$K_{ss}$")
    ax1a.plot(tau_plot * 100, Yss_plot, "r-", linewidth=2, label="$Y_{ss}$")
    ax1a.plot(tau_plot * 100, Css_plot, "g-", linewidth=2, label="$C_{ss}$")
    for tau_k in tau_values:
        sol = solutions[tau_k]
        ax1a.plot(tau_k * 100, sol["Kss"], "bo", markersize=8)
        ax1a.plot(tau_k * 100, sol["Yss"], "rs", markersize=8)
        ax1a.plot(tau_k * 100, sol["Css"], "g^", markersize=8)
    ax1a.set_xlabel("Capital tax rate $\\tau_k$ (%)")
    ax1a.set_ylabel("Steady-state level")
    ax1a.set_title("Steady State vs Tax Rate")
    ax1a.legend()

    # Percentage loss relative to zero tax
    ax1b.plot(tau_plot * 100, (Kss_plot / Kss_notax - 1) * 100, "b-", linewidth=2, label="$K_{ss}$")
    ax1b.plot(tau_plot * 100, (Yss_plot / Yss_notax - 1) * 100, "r-", linewidth=2, label="$Y_{ss}$")
    ax1b.plot(tau_plot * 100, (Css_plot / Css_notax - 1) * 100, "g-", linewidth=2, label="$C_{ss}$")
    ax1b.axhline(0, color="k", linewidth=0.5)
    ax1b.set_xlabel("Capital tax rate $\\tau_k$ (%)")
    ax1b.set_ylabel("% change from zero-tax SS")
    ax1b.set_title("Welfare Cost of Capital Taxation")
    ax1b.legend()
    fig1.tight_layout()
    report.add_figure("figures/steady-state-tax.png", "Steady-state levels and percentage losses as a function of the capital tax rate", fig1,
        description="The left panel shows how the steady-state capital stock falls steeply with the tax "
        "rate, dragging output down with it. The right panel reveals that capital losses are roughly "
        "twice the output losses due to the capital-share amplification: K falls as (1-tau)^{1/(1-alpha)}, "
        "a highly nonlinear relationship.")

    # --- Figure 2: Policy functions across tax rates ---
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13, 5))
    colors_tax = plt.cm.coolwarm(np.linspace(0.0, 1.0, len(tau_values)))

    iz_mid = 2  # median TFP state (index for n_z=5)
    for i, tau_k in enumerate(tau_values):
        sol = solutions[tau_k]
        ax2a.plot(sol["K_grid"], sol["policy_k"][iz_mid, :], color=colors_tax[i],
                  linewidth=2, label=f"$\\tau_k$={tau_k:.0%}")
        ax2b.plot(sol["K_grid"], sol["policy_c"][iz_mid, :], color=colors_tax[i],
                  linewidth=2, label=f"$\\tau_k$={tau_k:.0%}")

    ax2a.set_xlabel("Capital $K$")
    ax2a.set_ylabel("$K'$")
    ax2a.set_title("Capital Policy (median TFP)")
    ax2a.legend(fontsize=8)

    ax2b.set_xlabel("Capital $K$")
    ax2b.set_ylabel("Consumption $c$")
    ax2b.set_title("Consumption Policy (median TFP)")
    ax2b.legend(fontsize=8)
    fig2.tight_layout()
    report.add_figure("figures/policy-by-tax.png", "Policy functions at median TFP for different capital tax rates", fig2,
        description="Each curve shows the optimal decision rule for a different tax regime. Higher taxes "
        "shift the capital policy downward (less saving) and the consumption policy upward (more current "
        "consumption), because the after-tax return on capital no longer justifies the same level of "
        "investment.")

    # --- Figure 3: Simulated capital paths ---
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5))
    t_plot = slice(burn, burn + 300)
    for i, tau_k in enumerate(tau_values):
        sim = simulations[tau_k]
        ax3a.plot(sim["K"][t_plot], color=colors_tax[i], linewidth=0.8, alpha=0.9,
                  label=f"$\\tau_k$={tau_k:.0%}")
        ax3b.plot(sim["Y"][t_plot], color=colors_tax[i], linewidth=0.8, alpha=0.9,
                  label=f"$\\tau_k$={tau_k:.0%}")

    ax3a.set_xlabel("Period")
    ax3a.set_ylabel("Capital $K$")
    ax3a.set_title("Simulated Capital Paths")
    ax3a.legend(fontsize=7)

    ax3b.set_xlabel("Period")
    ax3b.set_ylabel("Output $Y$")
    ax3b.set_title("Simulated Output Paths")
    ax3b.legend(fontsize=7)
    fig3.tight_layout()
    report.add_figure("figures/simulation-paths.png", "Simulated capital and output paths under different tax rates", fig3,
        description="The simulations share identical TFP shock sequences, so differences across colors are "
        "purely due to the tax distortion. Notice how higher-tax economies (warmer colors) operate at "
        "permanently lower capital and output levels, illustrating the long-run cost of capital taxation.")

    # --- Figure 4: Investment response to TFP shock ---
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 5))
    for i, tau_k in enumerate(tau_values):
        sim = simulations[tau_k]
        K_data = sim["K"][burn:]
        I_data = sim["I"][burn:]
        Y_data = sim["Y"][burn:]

        # Distribution of investment rates
        inv_rate = I_data / Y_data
        ax4a.hist(inv_rate, bins=40, alpha=0.4, color=colors_tax[i], density=True,
                  label=f"$\\tau_k$={tau_k:.0%}")

    ax4a.set_xlabel("Investment rate $I/Y$")
    ax4a.set_ylabel("Density")
    ax4a.set_title("Distribution of Investment Rate")
    ax4a.legend(fontsize=7)

    # Capital-output ratio
    for i, tau_k in enumerate(tau_values):
        sol = solutions[tau_k]
        sim = simulations[tau_k]
        ky = sim["K"][burn:] / sim["Y"][burn:]
        ax4b.hist(ky, bins=40, alpha=0.4, color=colors_tax[i], density=True,
                  label=f"$\\tau_k$={tau_k:.0%}")
    ax4b.set_xlabel("Capital-output ratio $K/Y$")
    ax4b.set_ylabel("Density")
    ax4b.set_title("Distribution of Capital-Output Ratio")
    ax4b.legend(fontsize=7)
    fig4.tight_layout()
    report.add_figure("figures/investment-distributions.png", "Distribution of investment rate and capital-output ratio across tax regimes", fig4,
        description="Higher taxes compress the investment-rate distribution toward lower values (left) "
        "and reduce the capital-output ratio (right). The shift in these ratios is the core mechanism "
        "through which capital taxes affect growth: a persistently lower investment share erodes the "
        "economy's productive capacity.")

    # --- Table ---
    df_ss = pd.DataFrame(ss_data)
    report.add_table("tables/steady-state.csv", "Steady State and Simulation Statistics by Tax Rate", df_ss,
        description="The K_ss/K_ss(0) column shows how each tax rate erodes the capital stock relative to "
        "the zero-tax benchmark. Compare the analytical steady-state K_ss to the simulated mean K: the "
        "gap reflects precautionary savings, which partly offsets the tax distortion.")

    report.add_results(
        f"A 30% capital tax reduces steady-state capital by "
        f"{(1 - solutions[0.30]['Kss']/Kss_notax)*100:.1f}% and steady-state output by "
        f"{(1 - solutions[0.30]['Yss']/Yss_notax)*100:.1f}% relative to the no-tax benchmark. "
        f"The consumption loss is smaller ({(1 - solutions[0.30]['Css']/Css_notax)*100:.1f}%) "
        f"because reduced capital also means less depreciation.\n\n"
        f"Higher taxes shift the entire capital policy function downward: for any given state, "
        f"the agent chooses less capital accumulation because the after-tax return is lower. "
        f"This creates a permanently lower capital stock and output level."
    )

    report.add_takeaway(
        "Capital taxation has powerful long-run effects through the accumulation channel:\n\n"
        "1. **Steady-state distortion**: The tax-adjusted Euler equation $K_{ss}(\\tau) \\propto "
        "(1-\\tau)^{1/(1-\\alpha)}$ shows capital falls more than proportionally with the tax rate "
        "due to the capital share amplification.\n\n"
        "2. **Laffer curve in levels**: While tax revenue rises initially, the eroding base means "
        "that very high capital taxes can actually reduce total revenue.\n\n"
        "3. **Business cycle interaction**: Higher taxes reduce the investment-output ratio, making "
        "consumption a larger share of output. This can reduce output volatility (consumption is "
        "smoother than investment) but increase welfare costs of fluctuations.\n\n"
        "4. **Dynamic inefficiency**: The capital tax drives a wedge between the social and private "
        "return on capital, causing the economy to underaccumulate capital relative to the optimum."
    )

    report.add_references([
        "Chamley, C. (1986). *Optimal Taxation of Capital Income in General Equilibrium*. Econometrica.",
        "Judd, K. (1985). *Redistributive Taxation in a Simple Perfect Foresight Model*. JPE.",
        "Cao, D., Luo, W., and Nie, G. (2023). *Global DSGE Models*. Review of Economic Dynamics.",
        "Cole, H. and Obstfeld, M. (1991). *Commodity Trade and International Risk Sharing*. JME.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
