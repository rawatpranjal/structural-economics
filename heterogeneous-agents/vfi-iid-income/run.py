#!/usr/bin/env python3
"""VFI with IID Income Risk: Precautionary Savings Under Uncertainty.

Solves the infinite-horizon consumption-savings problem with IID income shocks
using value function iteration. Agents face uninsurable income risk and choose
how much to save in a risk-free asset subject to a borrowing constraint.

Reference: Greg Kaplan (2017), Heterogeneous Agent Models lecture notes.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy.optimize import fsolve

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def discrete_normal_np(n, mu, sigma, width):
    """Equally spaced approximation to normal distribution (pure NumPy).

    Returns:
        error: Approximation error in standard deviation.
        grid: (n, 1) array of grid points.
        probs: (n, 1) array of probabilities.
    """
    x = np.linspace(mu - width * sigma, mu + width * sigma, n).reshape(n, 1)
    if n == 2:
        p = 0.5 * np.ones((n, 1))
    else:
        p = np.zeros((n, 1))
        p[0] = norm.cdf(x[0, 0] + 0.5 * (x[1, 0] - x[0, 0]), mu, sigma)
        for i in range(1, n - 1):
            p[i] = (norm.cdf(x[i, 0] + 0.5 * (x[i + 1, 0] - x[i, 0]), mu, sigma)
                     - norm.cdf(x[i, 0] - 0.5 * (x[i, 0] - x[i - 1, 0]), mu, sigma))
        p[n - 1] = 1 - np.sum(p[:n - 1])

    ex = float((x.T @ p)[0, 0])
    sdx = float(np.sqrt(((x.T ** 2) @ p - ex ** 2)[0, 0]))
    error = sdx - sigma
    return error, x, p


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    # Preferences
    risk_aver = 2       # CRRA coefficient
    beta = 0.95         # Discount factor

    # Returns
    r = 0.03            # Interest rate
    R = 1 + r           # Gross return

    # Income risk: discretized N(mu, sigma^2)
    mu_y = 1.0          # Mean income
    sd_y = 0.2          # Std dev of income
    ny = 5              # Number of income grid points

    # Asset grid
    na = 1000           # Asset grid points
    amax = 20.0         # Maximum assets
    borrow_lim = 0.0    # Borrowing limit

    # Computation
    max_iter = 1000
    tol_iter = 1.0e-6

    # Simulation
    Nsim = 100          # Number of simulated agents
    Tsim = 500          # Simulation periods

    # =========================================================================
    # Random draws (fixed seed for reproducibility)
    # =========================================================================
    np.random.seed(2024)
    yrand = np.random.rand(Nsim, Tsim)

    # =========================================================================
    # Grids
    # =========================================================================
    # Asset grid (linear)
    agrid = np.linspace(borrow_lim, amax, na).reshape(na, 1)

    # Income grid: discretize normal distribution
    # Use scipy fsolve to find the optimal width parameter
    def width_objective(w):
        w_scalar = float(np.asarray(w).flat[0])
        err, _, _ = discrete_normal_np(ny, mu_y, sd_y, w_scalar)
        return err

    width = float(fsolve(width_objective, 2.0)[0])
    _, ygrid, ydist = discrete_normal_np(ny, mu_y, sd_y, width)
    ycumdist = np.cumsum(ydist)

    print(f"Income grid: {ygrid.flatten()}")
    print(f"Income probs: {ydist.flatten()}")
    print(f"Width parameter: {width:.4f}")

    # =========================================================================
    # Utility function
    # =========================================================================
    if risk_aver == 1:
        u = lambda c: np.log(c)
    else:
        u = lambda c: (c ** (1 - risk_aver) - 1) / (1 - risk_aver)

    # =========================================================================
    # Initialize value function
    # =========================================================================
    V = np.zeros((na, ny))
    for iy in range(ny):
        V[:, iy] = u(r * agrid[:, 0] + ygrid[iy, 0]) / (1 - beta)

    # =========================================================================
    # Value Function Iteration
    # =========================================================================
    print("\n--- Value Function Iteration ---")
    for iteration in range(1, max_iter + 1):
        Vlast = V.copy()
        V = np.zeros((na, ny))
        sav = np.zeros((na, ny))
        savind = np.zeros((na, ny), dtype=int)
        con = np.zeros((na, ny))

        # Expected continuation value: EV(a') = sum_y' V(a', y') * prob(y')
        EV = Vlast @ ydist  # (na, 1)

        for ia in range(na):
            for iy in range(ny):
                cash = R * agrid[ia, 0] + ygrid[iy, 0]
                # Consumption for each possible savings choice
                c_candidate = cash - agrid[:, 0]
                # Value of each savings choice
                Vchoice = u(np.maximum(c_candidate, 1.0e-10)) + beta * EV[:, 0]
                V[ia, iy] = np.max(Vchoice)
                savind[ia, iy] = np.argmax(Vchoice)
                sav[ia, iy] = agrid[savind[ia, iy], 0]
                con[ia, iy] = cash - sav[ia, iy]

        error = np.max(np.abs(V - Vlast))
        if iteration % 25 == 0:
            print(f"  VFI iteration {iteration:4d}, error = {error:.2e}")

        if error < tol_iter:
            print(f"  VFI converged in {iteration} iterations (error = {error:.2e})")
            break

    info = {"iterations": iteration, "converged": error < tol_iter, "error": error}

    # =========================================================================
    # Simulate
    # =========================================================================
    print("\n--- Simulation ---")
    yindsim = np.zeros((Nsim, Tsim), dtype=int)
    aindsim = np.zeros((Nsim, Tsim), dtype=int)

    # Initial assets: start at zero (borrowing constraint)
    aindsim[:, 0] = 0

    for it in range(Tsim):
        if (it + 1) % 100 == 0:
            print(f"  Simulating period {it + 1}/{Tsim}")

        # Income realization from IID draws
        yindsim[yrand[:, it] <= ycumdist[0], it] = 0
        for iy in range(1, ny):
            yindsim[
                np.logical_and(
                    yrand[:, it] > ycumdist[iy - 1],
                    yrand[:, it] <= ycumdist[iy],
                ),
                it,
            ] = iy

        # Asset choice for next period
        if it < Tsim - 1:
            for iy in range(ny):
                mask = yindsim[:, it] == iy
                aindsim[mask, it + 1] = savind[aindsim[mask, it], iy]

    # Map indices to values
    asim = agrid[aindsim, 0]  # (Nsim, Tsim)
    ysim = ygrid[yindsim, 0]  # (Nsim, Tsim)
    csim = np.zeros_like(asim)
    for it in range(Tsim):
        for i in range(Nsim):
            csim[i, it] = con[aindsim[i, it], yindsim[i, it]]

    # Summary statistics from final period
    a_final = asim[:, -1]
    y_final = ysim[:, -1]
    ay_final = a_final / np.mean(y_final)
    mean_assets = np.mean(ay_final)
    frac_constrained = np.sum(a_final == borrow_lim) / Nsim * 100
    pct_10 = np.quantile(ay_final, 0.1)
    pct_50 = np.quantile(ay_final, 0.5)
    pct_90 = np.quantile(ay_final, 0.9)
    pct_99 = np.quantile(ay_final, 0.99)

    print(f"\n--- Simulation Statistics (final period) ---")
    print(f"  Mean assets (relative to mean income): {mean_assets:.3f}")
    print(f"  Fraction at borrowing constraint: {frac_constrained:.1f}%")
    print(f"  10th percentile: {pct_10:.3f}")
    print(f"  50th percentile: {pct_50:.3f}")
    print(f"  90th percentile: {pct_90:.3f}")
    print(f"  99th percentile: {pct_99:.3f}")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "VFI with IID Income Risk",
        "Consumption-savings problem with uninsurable IID income shocks and a borrowing constraint.",
    )

    report.add_overview(
        "This model solves the canonical incomplete-markets consumption-savings problem "
        "where agents face IID income risk. Each period, the agent receives a random "
        "income draw from a discretized normal distribution and must decide how much to "
        "consume and how much to save in a risk-free asset.\n\n"
        "Unlike the deterministic case, agents face *uninsurable* income risk: they cannot "
        "write state-contingent contracts. This creates a **precautionary savings motive** -- "
        "agents save more than they would under certainty as a buffer against bad income "
        "realizations. The borrowing constraint further amplifies this motive by preventing "
        "agents from smoothing consumption via debt."
    )

    report.add_equations(
        r"""
$$V(a, y) = \max_{c \ge 0} \left\{ u(c) + \beta \, \mathbb{E}\left[ V(a', y') \right] \right\}$$

subject to:

$$a' = R \cdot a + y - c, \qquad a' \ge \underline{a}$$

where $a$ is assets, $y$ is income (IID), $R = 1+r$ is the gross interest rate,
and $\underline{a}$ is the borrowing limit.

**CRRA utility:** $u(c) = \frac{c^{1-\gamma}}{1-\gamma}$

**IID income:** $y \sim \mathcal{N}(\mu, \sigma^2)$, discretized to 5 points.

Since income is IID, the expectation simplifies:
$$\mathbb{E}[V(a', y')] = \sum_{j=1}^{n_y} V(a', y_j) \cdot \pi_j$$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\gamma$ | {risk_aver} | CRRA risk aversion |\n"
        f"| $\\beta$ | {beta} | Discount factor |\n"
        f"| $r$ | {r} | Interest rate |\n"
        f"| $\\mu_y$ | {mu_y} | Mean income |\n"
        f"| $\\sigma_y$ | {sd_y} | Std dev of income |\n"
        f"| $n_y$ | {ny} | Income grid points |\n"
        f"| $n_a$ | {na} | Asset grid points |\n"
        f"| $a_{{\\max}}$ | {amax} | Maximum assets |\n"
        f"| $\\underline{{a}}$ | {borrow_lim} | Borrowing limit |\n"
        f"| $N_{{sim}}$ | {Nsim} | Simulated agents |\n"
        f"| $T_{{sim}}$ | {Tsim} | Simulation periods |"
    )

    report.add_solution_method(
        "**Value Function Iteration (VFI):** We iterate on the Bellman equation:\n\n"
        "$$V_{n+1}(a, y) = \\max_{a' \\ge \\underline{a}} "
        "\\left\\{ u(Ra + y - a') + \\beta \\sum_{j} V_n(a', y_j) \\pi_j \\right\\}$$\n\n"
        "until $\\|V_{n+1} - V_n\\|_\\infty < 10^{-6}$. Because income is IID, the "
        "expected continuation value $\\mathbb{E}[V(a', y')]$ depends only on $a'$ "
        "(not the current income state), which simplifies computation.\n\n"
        "We search over the asset grid for the optimal savings choice at each state "
        "$(a, y)$, exploiting the fact that consumption is residual: $c = Ra + y - a'$.\n\n"
        f"Converged in **{info['iterations']} iterations** (error = {info['error']:.2e})."
    )

    # --- Figure 1: Value Functions ---
    fig1, ax1 = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, ny))
    for iy in range(ny):
        ax1.plot(
            agrid[:, 0], V[:, iy],
            color=colors[iy], linewidth=2,
            label=f"$y = {ygrid[iy, 0]:.2f}$",
        )
    ax1.set_xlabel("Assets $a$")
    ax1.set_ylabel("$V(a, y)$")
    ax1.set_title("Value Functions by Income State")
    ax1.legend()
    ax1.set_xlim(0, 5)
    report.add_figure(
        "figures/value-functions.png",
        "Value functions for each income state -- higher income shifts V up",
        fig1,
        description="Higher current income shifts the value function up because the agent has more "
        "resources available. The gap between curves narrows at high asset levels, where wealth "
        "dominates current income in determining lifetime welfare.",
    )

    # --- Figure 2: Consumption Policy ---
    fig2, ax2 = plt.subplots()
    for iy in [0, ny // 2, ny - 1]:
        ax2.plot(
            agrid[:, 0], con[:, iy],
            linewidth=2,
            label=f"$y = {ygrid[iy, 0]:.2f}$",
        )
    ax2.plot(agrid[:, 0], agrid[:, 0], "k:", linewidth=0.8, alpha=0.5, label="45-degree line")
    ax2.set_xlabel("Assets $a$")
    ax2.set_ylabel("Consumption $c$")
    ax2.set_title("Consumption Policy Function")
    ax2.legend()
    ax2.set_xlim(0, 5)
    report.add_figure(
        "figures/consumption-policy.png",
        "Consumption policy: agents with higher income consume more at every asset level",
        fig2,
        description="The consumption function is concave in assets, reflecting the precautionary "
        "savings motive: low-wealth agents have a high marginal propensity to consume, while "
        "wealthy agents save most of any additional dollar.",
    )

    # --- Figure 3: Savings Policy ---
    fig3, ax3 = plt.subplots()
    for iy in [0, ny // 2, ny - 1]:
        ax3.plot(
            agrid[:, 0], sav[:, iy] - agrid[:, 0],
            linewidth=2,
            label=f"$y = {ygrid[iy, 0]:.2f}$",
        )
    ax3.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    ax3.set_xlabel("Assets $a$")
    ax3.set_ylabel("Net savings $a' - a$")
    ax3.set_title("Savings Policy Function")
    ax3.legend()
    ax3.set_xlim(0, 5)
    report.add_figure(
        "figures/savings-policy.png",
        "Net savings: low-income agents dissave, high-income agents accumulate",
        fig3,
        description="The zero crossing of each curve marks the target asset level for that income "
        "state. Unlike the deterministic model, the target depends on current income: agents "
        "hit by low income run down their buffer stock, while those with high income rebuild it.",
    )

    # --- Figure 4: Simulated Asset Paths ---
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Individual paths (first 10 agents)
    n_show = min(10, Nsim)
    for i in range(n_show):
        ax4a.plot(range(Tsim), asim[i, :], linewidth=0.5, alpha=0.6)
    ax4a.plot(range(Tsim), np.mean(asim, axis=0), "k-", linewidth=2, label="Mean")
    ax4a.set_xlabel("Period")
    ax4a.set_ylabel("Assets $a_t$")
    ax4a.set_title("Simulated Asset Paths")
    ax4a.legend()

    # Panel B: Mean asset convergence
    ax4b.plot(range(Tsim), np.mean(asim, axis=0), "b-", linewidth=2)
    ax4b.set_xlabel("Period")
    ax4b.set_ylabel("Mean assets")
    ax4b.set_title("Mean Asset Convergence")
    fig4.tight_layout()
    report.add_figure(
        "figures/simulated-paths.png",
        "Simulated asset paths and mean convergence across agents",
        fig4,
        description="Individual paths fluctuate around the buffer stock target as agents are hit "
        "by income shocks, but the cross-sectional mean converges to a stationary level. "
        "The dispersion across paths reflects the wealth inequality generated by uninsurable risk.",
    )

    # --- Table: Policy values at selected grid points ---
    # Pick asset levels that span the interesting range
    a_select = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
    a_indices = [np.argmin(np.abs(agrid[:, 0] - a)) for a in a_select]

    table_data = {
        "Assets (a)": [f"{agrid[i, 0]:.2f}" for i in a_indices],
    }
    # Add consumption policy for lowest, middle, and highest income
    for iy, label in [(0, "Low y"), (ny // 2, "Mid y"), (ny - 1, "High y")]:
        table_data[f"c*(a, {label})"] = [f"{con[i, iy]:.4f}" for i in a_indices]
    for iy, label in [(0, "Low y"), (ny // 2, "Mid y"), (ny - 1, "High y")]:
        table_data[f"a'(a, {label})"] = [f"{sav[i, iy]:.4f}" for i in a_indices]

    df = pd.DataFrame(table_data)
    report.add_table(
        "tables/policy-values.csv",
        "Consumption and Savings Policy at Selected Asset Grid Points",
        df,
        description="Compare consumption across income states at each asset level: the gap "
        "between high-income and low-income consumption shrinks as assets increase, showing "
        "that wealthy agents can better smooth consumption across income realizations.",
    )

    # --- Simulation statistics table ---
    stats_data = {
        "Statistic": [
            "Mean assets / mean income",
            "Fraction at borrowing constraint",
            "10th percentile",
            "50th percentile (median)",
            "90th percentile",
            "99th percentile",
        ],
        "Value": [
            f"{mean_assets:.3f}",
            f"{frac_constrained:.1f}%",
            f"{pct_10:.3f}",
            f"{pct_50:.3f}",
            f"{pct_90:.3f}",
            f"{pct_99:.3f}",
        ],
    }
    df_stats = pd.DataFrame(stats_data)
    report.add_table(
        "tables/simulation-stats.csv",
        "Cross-Sectional Asset Distribution (Final Period)",
        df_stats,
        description="The right-skewed distribution is evident from the gap between mean and median "
        "wealth. Even with IID income shocks, a nontrivial fraction of agents are at the "
        "borrowing constraint, unable to smooth consumption against bad draws.",
    )

    report.add_takeaway(
        "IID income risk fundamentally changes savings behavior compared to the "
        "deterministic case.\n\n"
        "**Key insights:**\n"
        "- **Precautionary savings:** Agents save *more* than they would under certainty "
        "as a buffer against bad income draws. The concavity of the value function (risk "
        "aversion) means agents are more hurt by low consumption than they are helped by "
        "high consumption, so they self-insure through asset accumulation.\n"
        "- **Borrowing constraint binds:** A positive fraction of agents are at the "
        "borrowing limit in any period. These constrained agents cannot smooth consumption "
        "when hit by low income, creating welfare losses.\n"
        "- **Wealth inequality:** Even with IID (no persistent) income shocks, the "
        "stationary asset distribution is right-skewed -- a few agents accumulate "
        "substantial wealth while many remain near the constraint.\n"
        "- **IID simplification:** Because income is IID, the expected continuation value "
        "$\\mathbb{E}[V(a', y')]$ depends only on $a'$, not the current income state. "
        "This makes the problem computationally simpler than the AR(1) case where the "
        "full state is $(a, y)$ with persistent transitions."
    )

    report.add_references([
        "Kaplan, G. (2017). *Heterogeneous Agent Models: Lecture Notes*.",
        "Deaton, A. (1991). Saving and Liquidity Constraints. *Econometrica*, 59(5), 1221-1248.",
        "Carroll, C. (1997). Buffer-Stock Saving and the Life Cycle/Permanent Income Hypothesis. "
        "*Quarterly Journal of Economics*, 112(1), 1-55.",
        "Aiyagari, S.R. (1994). Uninsured Idiosyncratic Risk and Aggregate Saving. "
        "*Quarterly Journal of Economics*, 109(3), 659-684.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
