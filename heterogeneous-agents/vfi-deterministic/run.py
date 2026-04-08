#!/usr/bin/env python3
"""Deterministic VFI Consumption-Savings Model.

Solves the infinite-horizon consumption-savings problem with deterministic income
using value function iteration. A single agent chooses how much to save on a
discrete asset grid, subject to a borrowing constraint, earning a fixed interest
rate on savings and receiving constant income each period.

Reference: Kaplan (2017), HA Codes; Ljungqvist and Sargent (2018), Ch. 16.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    ## preferences
    risk_aver = 2       # CRRA coefficient
    beta = 0.95         # Discount factor

    ## returns
    r = 0.03            # Interest rate
    R = 1 + r           # Gross return

    ## income
    y = 1               # Deterministic income per period

    ## asset grid
    na = 1000           # Number of asset grid points
    amax = 20           # Maximum assets
    borrow_lim = 0      # Borrowing limit (natural: no debt)
    agrid_par = 1       # Grid curvature (1 = linear)

    ## computation
    max_iter = 1000     # Maximum VFI iterations
    tol_iter = 1.0e-6   # Convergence tolerance

    ## simulation
    Nsim = 100          # Number of simulated agents
    Tsim = 500          # Simulation periods

    # =========================================================================
    # Asset Grid
    # =========================================================================
    agrid = np.linspace(0, 1, na)
    agrid = agrid ** (1 / agrid_par)
    agrid = borrow_lim + (amax - borrow_lim) * agrid

    # =========================================================================
    # Utility Function
    # =========================================================================
    if risk_aver == 1:
        u = lambda c: np.log(c)
    else:
        u = lambda c: (c ** (1 - risk_aver) - 1) / (1 - risk_aver)

    # =========================================================================
    # Initialize Value Function
    # =========================================================================
    Vguess = u(r * agrid + y) / (1 - beta)
    V = Vguess.copy()

    # =========================================================================
    # Value Function Iteration
    # =========================================================================
    Vdiff = 1.0
    iteration = 0

    while iteration <= max_iter and Vdiff > tol_iter:
        iteration += 1
        Vlast = V.copy()
        V = np.zeros(na)
        sav = np.zeros(na)
        savind = np.zeros(na, dtype=int)
        con = np.zeros(na)

        for ia in range(na):
            cash = R * agrid[ia] + y
            Vchoice = u(np.maximum(cash - agrid, 1.0e-10)) + beta * Vlast
            V[ia] = np.max(Vchoice)
            savind[ia] = np.argmax(Vchoice)
            sav[ia] = agrid[savind[ia]]
            con[ia] = cash - sav[ia]

        Vdiff = np.max(np.abs(V - Vlast))
        if iteration % 50 == 0:
            print(f"  VFI iteration {iteration:4d}, max diff = {Vdiff:.2e}")

    converged = Vdiff <= tol_iter
    print(f"  VFI {'converged' if converged else 'did NOT converge'} in {iteration} iterations (error = {Vdiff:.2e})")

    # =========================================================================
    # Simulation
    # =========================================================================
    np.random.seed(2020)
    arand = np.random.rand(Nsim)

    aindsim = np.zeros((Nsim, Tsim), dtype=int)

    # Initial assets: uniform on [borrow_lim, amax], mapped to nearest grid point
    ainitial = borrow_lim + arand * (amax - borrow_lim)
    aindsim[:, 0] = interp1d(agrid, np.arange(na), kind='nearest', fill_value=(0, na - 1), bounds_error=False)(ainitial).astype(int)

    # Forward simulation using savings policy index
    for it in range(Tsim - 1):
        if (it + 1) % 100 == 0:
            print(f"  Simulating, time period {it + 1}")
        aindsim[:, it + 1] = savind[aindsim[:, it]]

    # Convert indices to levels
    asim = agrid[aindsim]
    csim = R * asim[:, :Tsim - 1] + y - asim[:, 1:Tsim]

    # Compute steady-state asset (where a' = a)
    ss_idx = np.argmin(np.abs(sav - agrid))
    a_ss = agrid[ss_idx]
    c_ss = R * a_ss + y - a_ss

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Deterministic VFI Consumption-Savings",
        "Infinite-horizon savings problem with deterministic income, solved by value function iteration on a discrete asset grid.",
    )

    report.add_overview(
        "This is the simplest heterogeneous-agents building block: a single agent choosing "
        "how much to consume and save each period, facing a constant interest rate $r$ and "
        "deterministic income $y$. The agent can save in a risk-free asset but faces a "
        "borrowing constraint $a \\ge 0$.\n\n"
        "Despite its simplicity, this model introduces the core mechanics of HA models: "
        "discrete asset grids, the consumption-savings Bellman equation, and forward simulation "
        "of asset dynamics. The borrowing constraint generates a kink in the consumption policy "
        "function at low wealth levels."
    )

    report.add_equations(
        r"""
$$V(a) = \max_{a' \in [\underline{a},\, \bar{a}]} \left\{ u(c) + \beta \, V(a') \right\}$$

subject to the budget constraint:

$$c = Ra + y - a'$$

where $a$ is current assets, $a'$ is savings (next-period assets), $R = 1+r$ is the gross
interest rate, and $y$ is deterministic income.

**CRRA utility:** $u(c) = \frac{c^{1-\gamma} - 1}{1 - \gamma}$, with $\gamma$ the coefficient
of relative risk aversion.

**Euler equation:** $u'(c) \ge \beta R \, u'(c')$, with equality when $a' > \underline{a}$.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\gamma$  | {risk_aver} | CRRA risk aversion |\n"
        f"| $\\beta$   | {beta} | Discount factor |\n"
        f"| $r$       | {r} | Interest rate |\n"
        f"| $y$       | {y} | Deterministic income |\n"
        f"| $\\underline{{a}}$ | {borrow_lim} | Borrowing limit |\n"
        f"| Grid points | {na} | Linear spacing on $[0, {amax}]$ |\n"
        f"| Simulated agents | {Nsim} | Forward simulation |\n"
        f"| Simulation periods | {Tsim} | Time horizon |"
    )

    report.add_solution_method(
        "**Value Function Iteration (VFI):** Starting from an initial guess "
        "$V_0(a) = u(ra + y) / (1 - \\beta)$, we iterate on the Bellman equation:\n\n"
        "$$V_{n+1}(a) = \\max_{a' \\in [\\underline{a},\\, \\bar{a}]} "
        "\\left\\{ u(Ra + y - a') + \\beta \\, V_n(a') \\right\\}$$\n\n"
        "The maximization is performed by evaluating all feasible savings choices on the "
        "asset grid (discrete search). Convergence is declared when "
        "$\\|V_{n+1} - V_n\\|_\\infty < 10^{-6}$.\n\n"
        f"Converged in **{iteration} iterations** (error = {Vdiff:.2e}).\n\n"
        f"**Steady-state assets:** $a^* = {a_ss:.4f}$, **steady-state consumption:** $c^* = {c_ss:.4f}$."
    )

    # --- Figure 1: Value Function ---
    fig1, ax1 = plt.subplots()
    ax1.plot(agrid, V, "b-", linewidth=2)
    ax1.set_xlabel("Assets $a$")
    ax1.set_ylabel("$V(a)$")
    ax1.set_title("Value Function")
    ax1.set_xlim(0, amax)
    report.add_figure("figures/value-function.png", "Value function V(a) over the asset grid", fig1,
        description="The value function is concave and increasing in assets, reflecting diminishing "
        "marginal value of wealth. The curvature is driven by CRRA preferences: an extra dollar "
        "matters much more to a poor agent than a rich one.")


    # --- Figure 2: Consumption Policy ---
    fig2, ax2 = plt.subplots()
    ax2.plot(agrid, con, "b-", linewidth=2, label="$c^*(a)$")
    ax2.plot(agrid, R * agrid + y, "k:", linewidth=0.8, alpha=0.5, label="Cash-on-hand $Ra + y$")
    ax2.set_xlabel("Assets $a$")
    ax2.set_ylabel("Consumption $c$")
    ax2.set_title("Consumption Policy Function")
    ax2.set_xlim(0, amax)
    ax2.legend()
    report.add_figure("figures/consumption-policy.png", "Consumption policy function c(a)", fig2,
        description="At low wealth the borrowing constraint binds and the agent consumes all "
        "cash-on-hand (the policy tracks the 45-degree line). As wealth increases, the agent "
        "saves a growing fraction of resources, and the consumption function flattens.")


    # --- Figure 3: Savings Policy ---
    fig3, ax3 = plt.subplots()
    ax3.plot(agrid, sav - agrid, "b-", linewidth=2, label="$a' - a$")
    ax3.axhline(0, color="k", linewidth=0.5)
    ax3.set_xlabel("Assets $a$")
    ax3.set_ylabel("Net savings $a' - a$")
    ax3.set_title("Savings Policy Function")
    ax3.set_xlim(0, amax)
    ax3.legend()
    report.add_figure("figures/savings-policy.png", "Savings policy: net change in assets a'-a as a function of current assets", fig3,
        description="The zero crossing identifies the steady-state asset level: agents below it "
        "accumulate wealth, agents above it decumulate. With beta*R < 1 (impatience dominates), "
        "the steady state is interior and all agents converge to it monotonically.")


    # --- Figure 4: Simulated Asset Dynamics ---
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    for i in range(min(Nsim, 20)):  # Plot a subset to avoid clutter
        ax4.plot(range(Tsim), asim[i, :], linewidth=0.5, alpha=0.6)
    ax4.axhline(a_ss, color="r", linewidth=1.5, linestyle="--", label=f"Steady state $a^* = {a_ss:.2f}$")
    ax4.set_xlabel("Period")
    ax4.set_ylabel("Assets $a_t$")
    ax4.set_title("Simulated Asset Dynamics (20 agents)")
    ax4.set_xlim(0, Tsim)
    ax4.legend()
    report.add_figure("figures/asset-dynamics.png", "Simulated asset paths converging to steady state", fig4,
        description="Regardless of initial wealth, every agent converges to the same steady state. "
        "This degenerate long-run distribution is the hallmark of the deterministic model and "
        "motivates the introduction of income risk to generate a non-trivial wealth distribution.")


    # --- Table: Policy Function at Selected Grid Points ---
    sample_idx = np.linspace(0, na - 1, 10, dtype=int)
    table_data = {
        "Assets (a)": [f"{agrid[i]:.3f}" for i in sample_idx],
        "Consumption c(a)": [f"{con[i]:.4f}" for i in sample_idx],
        "Savings a'(a)": [f"{sav[i]:.4f}" for i in sample_idx],
        "Net savings a'-a": [f"{(sav[i] - agrid[i]):.4f}" for i in sample_idx],
        "V(a)": [f"{V[i]:.4f}" for i in sample_idx],
    }
    df = pd.DataFrame(table_data)
    report.add_table("tables/policy-function.csv", "Policy Function at Selected Grid Points", df,
        description="At low asset levels, consumption nearly equals cash-on-hand and net savings "
        "are near zero (the constraint region). As assets grow, net savings become negative, "
        "confirming the agent dissaves toward the steady state from above.")


    report.add_takeaway(
        "The deterministic consumption-savings model illustrates how a borrowing-constrained "
        "agent accumulates assets toward a steady state.\n\n"
        "**Key insights:**\n"
        "- With $\\beta R < 1$ (impatience dominates returns), the agent has a finite steady-state "
        "asset level $a^*$ where consumption equals income plus net interest: $c^* = ra^* + y$.\n"
        "- The borrowing constraint $a \\ge 0$ binds for agents with very low wealth, creating "
        "a kink in the consumption policy where $c = Ra + y$ (consume everything).\n"
        "- All agents converge to the same steady state regardless of initial assets, since "
        "income is deterministic. This is why stochastic income (the next model) is needed "
        "to generate a non-degenerate wealth distribution.\n"
        "- The savings policy function $a' - a$ crosses zero exactly at the steady state: "
        "agents below save, agents above dissave."
    )

    report.add_references([
        "Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 16.",
        "Kaplan, G. (2017). *Heterogeneous Agent Models: Codes*. Lecture notes.",
        "Deaton, A. (1991). Saving and Liquidity Constraints. *Econometrica*, 59(5), 1221-1248.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
