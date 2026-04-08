#!/usr/bin/env python3
"""Consumption-Savings Problem with Markovian Income Shocks.

Solves the infinite-horizon income fluctuation problem (Bewley/Aiyagari) using
value function iteration. The agent chooses how much to consume and save each
period given stochastic income that follows an AR(1) process.

Reference: Ljungqvist and Sargent (2018), Ch. 18; Deaton (1991).
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.grids import exponential_grid
from lib.discretize import rouwenhorst
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    beta = 0.95          # Discount factor
    r = 0.03             # Risk-free interest rate
    sigma_crra = 2.0     # CRRA risk aversion coefficient
    borrowing_limit = 0.0  # Natural borrowing limit (no borrowing)

    # Income process: log(z') = rho * log(z) + eps, eps ~ N(0, sigma_eps^2)
    rho = 0.9            # Income persistence
    sigma_eps = 0.1      # Std dev of income innovation

    # Grids
    n_asset = 200        # Asset grid points
    a_min = borrowing_limit
    a_max = 20.0         # Maximum asset level
    n_income = 5         # Income states (Rouwenhorst)

    # VFI settings
    tol = 1e-6           # Convergence tolerance
    max_iter = 2000      # Maximum iterations

    # =========================================================================
    # CRRA Utility
    # =========================================================================
    def u(c):
        """CRRA utility (works with numpy arrays)."""
        c_safe = np.maximum(c, 1e-15)
        if sigma_crra == 1.0:
            return np.log(c_safe)
        else:
            return c_safe ** (1 - sigma_crra) / (1 - sigma_crra)

    def u_prime(c):
        """Marginal utility (for diagnostics)."""
        c_safe = np.maximum(c, 1e-15)
        return c_safe ** (-sigma_crra)

    # =========================================================================
    # Income Process: Rouwenhorst discretization of AR(1) in logs
    # =========================================================================
    # rouwenhorst(n, mu, sigma, rho) where sigma is std dev of innovation
    z_grid_jax, trans_jax, ergo_dist_jax = rouwenhorst(
        n=n_income, mu=0.0, sigma=sigma_eps, rho=rho
    )
    # z_grid is in log space (shape n_income x 1); convert to levels
    z_grid_log = np.array(z_grid_jax).flatten()   # log(z) values
    z_grid = np.exp(z_grid_log)                     # z values in levels
    trans = np.array(trans_jax)                      # (n_income, n_income)
    ergo_dist = np.array(ergo_dist_jax).flatten()

    print(f"Income grid (levels): {z_grid}")
    print(f"Ergodic distribution: {ergo_dist}")
    print(f"Transition matrix:\n{trans}")

    # =========================================================================
    # Asset Grid (exponential: denser near borrowing limit)
    # =========================================================================
    a_grid_jax = exponential_grid(a_min, a_max, n_asset, density=3.0)
    a_grid = np.array(a_grid_jax)

    # =========================================================================
    # Value Function Iteration
    # =========================================================================
    # State: (a, z) where a is assets and z is income
    # V(a, z) = max_{c} u(c) + beta * E[V(a', z') | z]
    # Budget: c + a' = (1+r)*a + z
    # Constraint: a' >= borrowing_limit, c >= 0

    # Initial guess: consume all cash-on-hand each period
    V = np.zeros((n_asset, n_income))
    for iz in range(n_income):
        for ia in range(n_asset):
            cash_on_hand = (1 + r) * a_grid[ia] + z_grid[iz]
            V[ia, iz] = u(cash_on_hand) / (1 - beta)

    # Precompute the fine grid for a' (next-period assets) = same as a_grid
    # For each state, do grid search over a' in a_grid
    a_prime_grid = a_grid.copy()
    n_aprime = len(a_prime_grid)

    print(f"\nStarting VFI with {n_asset} asset points x {n_income} income states...")

    for iteration in range(1, max_iter + 1):
        V_new = np.zeros((n_asset, n_income))
        policy_a_idx = np.zeros((n_asset, n_income), dtype=int)

        for iz in range(n_income):
            # Expected continuation value: E[V(a', z') | z] for each a'
            # EV(a') = sum_{z'} pi(z'|z) * V(a', z')
            EV = V @ trans[iz, :]  # shape (n_asset,) -- E[V(a', .) | z=z_iz]

            for ia in range(n_asset):
                cash_on_hand = (1 + r) * a_grid[ia] + z_grid[iz]

                # Feasible a' values: a' <= cash_on_hand and a' >= borrowing_limit
                # Consumption: c = cash_on_hand - a'
                consumption = cash_on_hand - a_prime_grid
                feasible = consumption > 1e-10

                if not np.any(feasible):
                    # Edge case: only option is a'=borrowing_limit
                    V_new[ia, iz] = u(cash_on_hand - borrowing_limit) + beta * EV[0]
                    policy_a_idx[ia, iz] = 0
                    continue

                values = np.full(n_aprime, -1e20)
                values[feasible] = u(consumption[feasible]) + beta * EV[feasible]

                best = np.argmax(values)
                V_new[ia, iz] = values[best]
                policy_a_idx[ia, iz] = best

        error = np.max(np.abs(V_new - V))
        if iteration % 50 == 0:
            print(f"  VFI iteration {iteration:4d}, error = {error:.2e}")
        V = V_new

        if error < tol:
            print(f"  VFI converged in {iteration} iterations (error = {error:.2e})")
            break
    else:
        print(f"  VFI did NOT converge after {max_iter} iterations (error = {error:.2e})")

    info = {"iterations": iteration, "converged": error < tol, "error": error}

    # =========================================================================
    # Extract Policy Functions
    # =========================================================================
    policy_a = np.zeros((n_asset, n_income))  # a'(a, z)
    policy_c = np.zeros((n_asset, n_income))  # c(a, z)

    for iz in range(n_income):
        for ia in range(n_asset):
            cash_on_hand = (1 + r) * a_grid[ia] + z_grid[iz]
            policy_a[ia, iz] = a_prime_grid[policy_a_idx[ia, iz]]
            policy_c[ia, iz] = cash_on_hand - policy_a[ia, iz]

    savings_policy = policy_a - a_grid[:, None]  # a' - a (net savings)

    # =========================================================================
    # Simulate Asset Paths
    # =========================================================================
    np.random.seed(42)
    T_sim = 200
    n_agents = 5

    # Simulate income paths using the Markov chain
    sim_assets = np.zeros((T_sim, n_agents))
    sim_income_idx = np.zeros((T_sim, n_agents), dtype=int)

    # Initialize agents at median income, zero assets
    median_z_idx = n_income // 2
    sim_income_idx[0, :] = median_z_idx
    sim_assets[0, :] = 0.0

    for t in range(T_sim - 1):
        for agent in range(n_agents):
            iz = sim_income_idx[t, agent]
            a_current = sim_assets[t, agent]

            # Interpolate policy function for a'(a, z)
            a_next = np.interp(a_current, a_grid, policy_a[:, iz])
            sim_assets[t + 1, agent] = np.clip(a_next, a_min, a_max)

            # Draw next income state
            cum_probs = np.cumsum(trans[iz, :])
            shock = np.random.rand()
            sim_income_idx[t + 1, agent] = np.searchsorted(cum_probs, shock)

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Consumption-Savings with Income Shocks",
        "Optimal saving and consumption under Markovian income uncertainty (Bewley/Aiyagari).",
    )

    report.add_overview(
        "The income fluctuation problem (also known as the Bewley or Aiyagari model at the "
        "individual level) studies how a risk-averse agent optimally allocates between "
        "consumption and savings when facing stochastic, mean-reverting income. A borrowing "
        "constraint prevents the agent from perfectly smoothing consumption, generating a "
        "precautionary savings motive.\n\n"
        "This is a foundational building block for heterogeneous-agent macroeconomics: "
        "aggregate the individual decision rules to obtain wealth distributions, and embed "
        "them in general equilibrium (Aiyagari 1994)."
    )

    report.add_equations(
        r"""
$$V(a, z) = \max_{c \ge 0} \left\{ u(c) + \beta \, \mathbb{E}\left[V(a', z') \mid z\right] \right\}$$

**Budget constraint:** $c + a' = (1+r) \, a + z$

**Borrowing constraint:** $a' \ge \underline{a}$

**CRRA utility:** $u(c) = \frac{c^{1-\sigma}}{1-\sigma}$

**Income process:** $\ln z' = \rho \, \ln z + \varepsilon$, $\quad \varepsilon \sim N(0, \sigma_\varepsilon^2)$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\beta$  | {beta} | Discount factor |\n"
        f"| $r$      | {r} | Risk-free interest rate |\n"
        f"| $\\sigma$ | {sigma_crra} | CRRA risk aversion |\n"
        f"| $\\rho$   | {rho} | Income persistence |\n"
        f"| $\\sigma_\\varepsilon$ | {sigma_eps} | Income shock std dev |\n"
        f"| $\\underline{{a}}$ | {borrowing_limit} | Borrowing limit |\n"
        f"| Asset grid | {n_asset} points | Exponential spacing on $[{a_min}, {a_max}]$ |\n"
        f"| Income states | {n_income} | Rouwenhorst discretization |"
    )

    report.add_solution_method(
        "**Value Function Iteration (VFI)** with discrete grid search over next-period "
        "assets $a'$. For each state $(a, z)$, the agent's cash-on-hand is "
        "$(1+r)a + z$, and we search over a grid of feasible $a'$ values to maximize "
        "current utility plus the discounted expected continuation value.\n\n"
        "The expected continuation value $\\mathbb{E}[V(a', z') | z]$ is computed by "
        "weighting $V(a', z')$ across income states using the Markov transition matrix.\n\n"
        f"Converged in **{info['iterations']} iterations** (error = {info['error']:.2e})."
    )

    # --- Figure 1: Value Functions ---
    fig1, ax1 = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_income))
    for iz in range(n_income):
        ax1.plot(a_grid, V[:, iz], color=colors[iz], linewidth=2,
                 label=f"$z = {z_grid[iz]:.3f}$")
    ax1.set_xlabel("Assets $a$")
    ax1.set_ylabel("$V(a, z)$")
    ax1.set_title("Value Functions by Income State")
    ax1.legend(fontsize=9)
    report.add_figure("figures/value-functions.png",
                       "Value functions for each income state", fig1,
        description="Higher income states shift the value function upward, reflecting greater lifetime utility. "
        "The curves fan out at low asset levels where the borrowing constraint binds, making current income more consequential.")

    # --- Figure 2: Consumption Policy Functions ---
    fig2, ax2 = plt.subplots()
    for iz in range(n_income):
        ax2.plot(a_grid, policy_c[:, iz], color=colors[iz], linewidth=2,
                 label=f"$z = {z_grid[iz]:.3f}$")
    ax2.plot(a_grid, (1 + r) * a_grid + z_grid[median_z_idx], "k:",
             linewidth=0.8, alpha=0.5, label="Cash-on-hand (median $z$)")
    ax2.set_xlabel("Assets $a$")
    ax2.set_ylabel("Consumption $c$")
    ax2.set_title("Consumption Policy Functions")
    ax2.legend(fontsize=9)
    report.add_figure("figures/consumption-policy.png",
                       "Consumption policy functions by income state", fig2,
        description="The concavity of the consumption function is the hallmark of precautionary savings: "
        "the marginal propensity to consume is highest near the borrowing constraint and declines with wealth, "
        "meaning windfall gains matter most for liquidity-constrained households.")

    # --- Figure 3: Savings Policy Functions ---
    fig3, ax3 = plt.subplots()
    for iz in range(n_income):
        ax3.plot(a_grid, savings_policy[:, iz], color=colors[iz], linewidth=2,
                 label=f"$z = {z_grid[iz]:.3f}$")
    ax3.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax3.set_xlabel("Assets $a$")
    ax3.set_ylabel("Net Savings $a' - a$")
    ax3.set_title("Savings Policy Functions")
    ax3.legend(fontsize=9)
    report.add_figure("figures/savings-policy.png",
                       "Net savings (a' - a) by income state", fig3,
        description="The zero-crossing of each curve identifies the target asset level for that income state. "
        "Low-income agents dissave (negative net savings) while high-income agents accumulate wealth, "
        "driving the mean-reverting asset dynamics that produce a stationary wealth distribution.")

    # --- Figure 4: Simulated Asset Paths ---
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sim_colors = plt.cm.Set1(np.linspace(0, 0.8, n_agents))
    for agent in range(n_agents):
        ax4.plot(np.arange(T_sim), sim_assets[:, agent], color=sim_colors[agent],
                 linewidth=1, alpha=0.8, label=f"Agent {agent+1}")
    ax4.set_xlabel("Period")
    ax4.set_ylabel("Assets $a_t$")
    ax4.set_title(f"Simulated Asset Paths ({n_agents} agents, {T_sim} periods)")
    ax4.legend(fontsize=9)
    report.add_figure("figures/simulated-paths.png",
                       f"Simulated asset paths for {n_agents} agents over {T_sim} periods", fig4,
        description="Despite identical preferences and the same income process, agents accumulate very different "
        "wealth levels over time due to idiosyncratic income realizations. This ex-post heterogeneity from ex-ante "
        "identical agents is the mechanism that generates the wealth distribution in the Aiyagari model.")

    # --- Table: Policy function at selected grid points ---
    # Sample at a few asset levels for the lowest, median, and highest income
    sample_a_idx = np.linspace(0, n_asset - 1, 8, dtype=int)
    iz_low, iz_mid, iz_high = 0, n_income // 2, n_income - 1
    table_data = {
        "Assets $a$": [f"{a_grid[i]:.2f}" for i in sample_a_idx],
        f"$c^*(a, z_{{low}})$": [f"{policy_c[i, iz_low]:.4f}" for i in sample_a_idx],
        f"$c^*(a, z_{{mid}})$": [f"{policy_c[i, iz_mid]:.4f}" for i in sample_a_idx],
        f"$c^*(a, z_{{high}})$": [f"{policy_c[i, iz_high]:.4f}" for i in sample_a_idx],
        f"$a'^*(a, z_{{low}})$": [f"{policy_a[i, iz_low]:.4f}" for i in sample_a_idx],
        f"$a'^*(a, z_{{mid}})$": [f"{policy_a[i, iz_mid]:.4f}" for i in sample_a_idx],
        f"$a'^*(a, z_{{high}})$": [f"{policy_a[i, iz_high]:.4f}" for i in sample_a_idx],
    }
    df = pd.DataFrame(table_data)
    report.add_table("tables/policy-functions.csv", "Policy Functions at Selected Grid Points", df,
        description="Comparing consumption and savings across income states reveals the precautionary motive: "
        "at any asset level, low-income agents save more relative to their resources than the certainty-equivalent benchmark would predict.")

    report.add_takeaway(
        "The income fluctuation problem reveals the **precautionary savings motive**: "
        "risk-averse agents save more than they would under certainty, building a buffer "
        "stock against bad income shocks.\n\n"
        "**Key insights:**\n"
        "- The consumption function is **concave** in wealth: poorer agents have a higher "
        "marginal propensity to consume (MPC) than wealthier agents. This creates the "
        "characteristic kink near the borrowing constraint.\n"
        "- Agents in **low income states** dissave (run down assets), while agents in "
        "**high income states** accumulate wealth. The net savings function crosses zero "
        "at different asset levels for each income state.\n"
        "- Higher income **persistence** ($\\rho$) amplifies wealth inequality: long runs "
        "of good or bad luck create large differences in accumulated assets.\n"
        "- The borrowing constraint binds for low-wealth, low-income agents, forcing them "
        "to consume less than they would under perfect markets. This is the key friction "
        "that drives precautionary behavior.\n"
        "- This individual problem is the building block of the Aiyagari (1994) general "
        "equilibrium model, where the interest rate $r$ adjusts to clear the capital market."
    )

    report.add_references([
        "Aiyagari, S. R. (1994). Uninsured Idiosyncratic Risk and Aggregate Saving. "
        "*Quarterly Journal of Economics*, 109(3), 659-684.",
        "Deaton, A. (1991). Saving and Liquidity Constraints. *Econometrica*, 59(5), 1221-1248.",
        "Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. "
        "MIT Press, 4th edition, Ch. 18.",
        "Carroll, C. D. (1997). Buffer-Stock Saving and the Life Cycle/Permanent Income "
        "Hypothesis. *Quarterly Journal of Economics*, 112(1), 1-55.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
