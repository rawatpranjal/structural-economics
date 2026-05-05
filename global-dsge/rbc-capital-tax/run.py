#!/usr/bin/env python3
"""Capital tax wedges in a stochastic RBC model.

Compares permanent capital income tax rates in a stochastic growth model. The
government taxes capital income and rebates the revenue lump-sum, so aggregate
resources are unchanged but the household Euler equation contains an after-tax
return. A global VFI pass initializes the policy; an Euler refinement enforces
the tax wedge.

References: Chamley (1986), Judd (1985), and the global DSGE examples in
Cao, Luo, and Nie (2023).
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
    """Solve the RBC model with permanent capital tax tau_k.

    Returns policy functions, value function, grids, and deterministic
    steady-state benchmarks.
    """
    # Steady state with tax: (1-tau_k) * alpha * K^(alpha-1) = 1/beta - 1 + delta
    Kss = ((1.0 - tau_k) * alpha / (1.0 / beta - 1.0 + delta)) ** (1.0 / (1.0 - alpha))
    Yss = Kss ** alpha
    Css = Yss - delta * Kss
    Iss = delta * Kss
    Tss = tau_k * alpha * Yss

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
    err_euler = np.inf
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
        "Kss": Kss, "Yss": Yss, "Css": Css, "Iss": Iss, "Tss": Tss,
        "K_min": K_min, "K_max": K_max,
        "tau_k": tau_k, "iterations": iteration,
        "euler_iterations": euler_iter + 1, "euler_error": err_euler,
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
        C_sim[t] = interp_c(pt).item()
        Y_sim[t] = z_sim[t] * K_sim[t] ** alpha
        kp = interp_k(pt).item()
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
            "T_ss": f"{sol['Tss']:.4f}",
            "K_ss / K_ss(0)": f"{sol['Kss']/Kss_notax:.3f}",
            "K loss %": f"{100*(1-sol['Kss']/Kss_notax):.1f}",
            "Mean K (sim)": f"{sim['K'][burn:].mean():.4f}",
            "std(Y) %": f"{100*np.std(np.log(sim['Y'][burn:]) - np.log(sim['Y'][burn:]).mean()):.3f}",
        })

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Capital Tax Wedges in an RBC Model",
        "A revenue-neutral capital tax leaves aggregate resources unchanged but lowers the after-tax return that governs saving.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Capital income taxation separates aggregate feasibility from "
        "private incentives. The government taxes capital income at rate $\\tau_k$ and "
        "rebates the proceeds lump-sum. The representative economy still has the same "
        "resource constraint, but households save against an after-tax marginal product "
        "of capital. The distortion is entirely intertemporal: current "
        "consumption becomes cheaper relative to future consumption.\n\n"
        "This tutorial is a tax-wedge companion to the global "
        "[RBC capital and labor](../../dynamic-programming/rbc/) example and the local "
        "[Dynare RBC](../../dynare/rbc/) impulse-response example. The point is not "
        "shock propagation per se, but how a permanent wedge moves the exact "
        "deterministic steady state, the nonlinear capital policy, and simulated "
        "investment behavior."
    )

    report.add_equations(
        r"""
Let $K_t$ be aggregate capital at the start of period $t$, $z_t$ aggregate
TFP, $c_t$ consumption, and $K_{t+1}$ next-period capital. Preferences are

$$\mathbb{E}_0 \sum_{t=0}^{\infty} \beta^t
\frac{c_t^{1-\sigma}}{1-\sigma}, \qquad \sigma>0,$$

with Cobb-Douglas output $Y_t=z_t K_t^\alpha$. Productivity follows

$$\log z_{t+1}=\rho \log z_t+\varepsilon_{t+1},
\qquad \varepsilon_{t+1}\sim N(0,\sigma_\varepsilon^2).$$

The government rebate means aggregate feasibility is the usual RBC resource
constraint,

$$c_t + K_{t+1} = z_t K_t^\alpha + (1-\delta)K_t.$$

The tax appears in the household Euler equation:

$$c_t^{-\sigma} =
\beta \mathbb{E}_t\left[
c_{t+1}^{-\sigma}
\left((1-\tau_k)\alpha z_{t+1}K_{t+1}^{\alpha-1}+1-\delta\right)
\right].$$

Thus the wedge changes the return to saving but not the goods available to the
economy in a given period.

At $z=1$, the exact deterministic steady state is

$$K_{ss}(\tau_k)=
\left(\frac{(1-\tau_k)\alpha}{1/\beta-1+\delta}\right)^{1/(1-\alpha)},$$

with $Y_{ss}=K_{ss}^{\alpha}$, $C_{ss}=Y_{ss}-\delta K_{ss}$, and
tax revenue $T_{ss}=\tau_k \alpha Y_{ss}$.
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
        f"| $\\tau_k$ | {tau_values} | Permanent tax rates compared |\n"
        f"| Capital grid | {40} points around each $K_{{ss}}(\\tau_k)$ | State and $K'$ choice grid |\n"
        f"| TFP grid | {5} Tauchen states | Approximation to log productivity |\n"
        f"| Simulation periods | {T_sim} | Same shock seed for every tax regime, with {burn} burn-in periods |"
    )

    report.add_solution_method(
        "The computation uses the resource-feasible Bellman problem to get a stable "
        "global policy on the $(z,K)$ grid, then refines consumption with the after-tax "
        "Euler equation. The first step is a good initializer because the rebate leaves "
        "the aggregate resource constraint unchanged. The second step introduces the "
        "capital-tax wedge.\n\n"
        "```text\n"
        "Algorithm: global policy iteration with a capital-tax wedge\n"
        "Input: tax rate tau_k, grids K and Z, transition matrix P, primitives beta, alpha, sigma, delta\n"
        "Output: value V(z,K), capital policy g_K(z,K), consumption policy g_c(z,K)\n"
        "Compute the exact deterministic K_ss(tau_k) and build a capital grid around it\n"
        "Discretize log productivity with Tauchen to obtain Z and P\n"
        "Precompute feasible consumption c = z K^alpha + (1-delta)K - K' for every (z,K,K')\n"
        "Initialize V_0(z,K)\n"
        "repeat:\n"
        "    for each state (z_i,K_m):\n"
        "        choose K' on the grid to maximize u(c) + beta * sum_j P_ij V_n(z_j,K')\n"
        "        record V_{n+1}, g_K, and g_c\n"
        "    apply Howard improvement to the fixed policy\n"
        "until the sup-norm value update is below epsilon\n"
        "repeat Euler refinement:\n"
        "    for each state (z_i,K_m):\n"
        "        K_plus = g_K(z_i,K_m)\n"
        "        M = sum_j P_ij g_c(z_j,K_plus)^(-sigma)\n"
        "            * ((1-tau_k) alpha z_j K_plus^(alpha-1) + 1-delta)\n"
        "        g_c_new(z_i,K_m) = (beta * M)^(-1/sigma)\n"
        "        g_K_new(z_i,K_m) = z_i K_m^alpha + (1-delta)K_m - g_c_new(z_i,K_m)\n"
        "until the consumption policy update is below epsilon\n"
        "Simulate all tax regimes on the same productivity path\n"
        "```\n\n"
        "The exact deterministic steady state serves as a ground-truth benchmark for "
        "the long-run comparisons. The stochastic policy functions are numerical, and "
        "the table below separates the exact steady states from simulated means. Across "
        f"the five tax regimes, VFI used at most **{max(sol['iterations'] for sol in solutions.values())}** "
        "outer iterations and the Euler refinement used at most "
        f"**{max(sol['euler_iterations'] for sol in solutions.values())}** iterations."
    )

    report.add_results(
        f"The exact steady-state formulas already show the size of the distortion. "
        f"At $\\tau_k=30\\%$, deterministic capital is "
        f"{(1 - solutions[0.30]['Kss']/Kss_notax)*100:.1f}% below the no-tax value, "
        f"output is {(1 - solutions[0.30]['Yss']/Yss_notax)*100:.1f}% lower, and "
        f"consumption is {(1 - solutions[0.30]['Css']/Css_notax)*100:.1f}% lower. "
        f"Consumption falls less because a lower capital stock also reduces replacement "
        f"investment. The simulations use the same productivity sequence for every tax "
        f"rate, so the level differences across paths are the tax wedge, not different "
        f"shock histories."
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
    ax1a.set_title("Exact steady states")
    ax1a.legend()

    # Percentage loss relative to zero tax
    ax1b.plot(tau_plot * 100, (Kss_plot / Kss_notax - 1) * 100, "b-", linewidth=2, label="$K_{ss}$")
    ax1b.plot(tau_plot * 100, (Yss_plot / Yss_notax - 1) * 100, "r-", linewidth=2, label="$Y_{ss}$")
    ax1b.plot(tau_plot * 100, (Css_plot / Css_notax - 1) * 100, "g-", linewidth=2, label="$C_{ss}$")
    ax1b.axhline(0, color="k", linewidth=0.5)
    ax1b.set_xlabel("Capital tax rate $\\tau_k$ (%)")
    ax1b.set_ylabel("% change from zero-tax steady state")
    ax1b.set_title("Exact steady-state losses")
    ax1b.legend()
    fig1.tight_layout()
    report.add_figure("figures/steady-state-tax.png", "Exact steady-state levels and losses by capital tax rate", fig1,
        description="The first comparison is analytical rather than simulated. Capital falls with "
        "$(1-\\tau_k)^{1/(1-\\alpha)}$, so the tax rate is amplified by the capital share. Output and "
        "consumption move less than capital, but the economy operates from a lower productive base.")

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
    report.add_figure("figures/policy-by-tax.png", "Capital and consumption policies at median TFP by capital tax rate", fig2,
        description="At the median productivity state, the policy functions show the same wedge in decision-rule form. "
        "Higher taxes move the capital policy down and the consumption policy up: the household chooses less saving "
        "because tomorrow's marginal product is partly taxed away.")

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
    report.add_figure("figures/simulation-paths.png", "Simulated capital and output paths by capital tax rate", fig3,
        description="The simulated paths keep the productivity sequence fixed across regimes. The higher-tax economies "
        "therefore track the same booms and recessions from permanently lower capital and output levels.")

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
    report.add_figure("figures/investment-distributions.png", "Investment-rate and capital-output distributions by tax regime", fig4,
        description="The distributional view is useful because the policy change is not only a new mean. Higher taxes "
        "shift the investment share and the capital-output ratio left across the stationary simulation, so the economy "
        "spends more time in states with a smaller productive base.")

    # --- Table ---
    df_ss = pd.DataFrame(ss_data)
    report.add_table("tables/steady-state.csv", "Exact Steady States and Simulated Moments by Tax Rate", df_ss,
        description="The table keeps the closed-form steady-state benchmark separate from the simulated mean. "
        "The simulated mean capital is slightly above the deterministic value because productivity risk and the "
        "nonlinear policy shift the invariant distribution, but the ranking across tax regimes is unchanged.")

    report.add_takeaway(
        "The rebate closes the government budget, not the intertemporal wedge. Once the household "
        "prices saving with $(1-\\tau_k)MPK$ rather than $MPK$, the economy carries less capital "
        "into every productivity state. The exact steady state is the cleanest way to see the "
        "long-run loss; the global policy functions show how the same force operates away from "
        "the steady state. The lesson for nearby DSGE applications is that fiscal wedges "
        "can be revenue-neutral in resources and still large in allocation."
    )

    report.add_references([
        "Chamley, C. (1986). *Optimal Taxation of Capital Income in General Equilibrium*. Econometrica.",
        "Judd, K. (1985). *Redistributive Taxation in a Simple Perfect Foresight Model*. JPE.",
        "Cao, D., Luo, W., and Nie, G. (2023). *Global DSGE Models*. Review of Economic Dynamics.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
