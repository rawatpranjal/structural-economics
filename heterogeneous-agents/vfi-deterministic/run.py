#!/usr/bin/env python3
"""Deterministic consumption-savings by grid VFI.

This tutorial solves the no-risk household problem that sits underneath many
incomplete-market models. With constant income and beta * R < 1, the economic
content is deliberately stark: absent income risk, an impatient household runs
assets down to the borrowing limit.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
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
    def utility(c):
        """CRRA utility evaluated only on positive consumption."""
        if risk_aver == 1:
            return np.log(c)
        return (c ** (1 - risk_aver) - 1) / (1 - risk_aver)

    # =========================================================================
    # Initialize Value Function
    # =========================================================================
    Vguess = utility(r * agrid + y) / (1 - beta)
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
            consumption_choices = cash - agrid
            feasible = consumption_choices > 0
            Vchoice = np.full(na, -np.inf)
            Vchoice[feasible] = (
                utility(consumption_choices[feasible]) + beta * Vlast[feasible]
            )
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
    exact_ss_a = borrow_lim
    exact_ss_c = r * exact_ss_a + y
    ss_error = abs(a_ss - exact_ss_a)
    beta_r = beta * R

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Deterministic Saving with a Borrowing Constraint",
        "A no-risk household savings problem solved by value function iteration on an asset grid.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "This tutorial strips the household side of a heterogeneous-agent model down to "
        "one state, one asset, and no income risk. The household receives the same income "
        "$y$ every period, saves in a risk-free asset with gross return $R=1+r$, and cannot "
        "borrow below $\\underline a=0$.\n\n"
        "That benchmark is useful precisely because it does not produce a meaningful wealth "
        "distribution. With $\\beta R<1$, patience is not strong enough to offset the return, "
        "so the household eventually runs any initial assets down to the borrowing limit. "
        "The next savings tutorials add income risk and faster Euler-equation methods; this "
        "one makes clear what is already present before those complications enter."
    )

    report.add_equations(
        r"""
For current assets $a \in [\underline a,\bar a]$, next-period assets $a'$ are chosen
from the same grid subject to positive consumption:

$$
V(a) =
\max_{a' \in [\underline a,\bar a]}
[u(c) + \beta V(a')],
\qquad
c = Ra + y - a' > 0.
$$

The utility function is CRRA,

$$
u(c)=
\begin{cases}
\dfrac{c^{1-\gamma}-1}{1-\gamma}, & \gamma \neq 1,\\[4pt]
\log c, & \gamma = 1.
\end{cases}
$$

Away from the borrowing limit, the Euler equation is

$$
u'(c_t) = \beta R u'(c_{t+1}).
$$

At $a'=\underline a$, the condition becomes an inequality. Since the calibration
has $\beta R<1$, there is no positive interior stationary asset level satisfying
the Euler equation; the exact stationary asset benchmark is $a^{*}=\underline a$.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\gamma$  | {risk_aver} | CRRA risk aversion |\n"
        f"| $\\beta$   | {beta} | Discount factor |\n"
        f"| $r$       | {r} | Interest rate |\n"
        f"| $\\beta R$ | {beta_r:.4f} | Patience-return product |\n"
        f"| $y$       | {y} | Deterministic income |\n"
        f"| $\\underline{{a}}$ | {borrow_lim} | Borrowing limit |\n"
        f"| Grid points | {na} | Linear spacing on $[0, {amax}]$ |\n"
        f"| Simulated agents | {Nsim} | Forward simulation |\n"
        f"| Simulation periods | {Tsim} | Time horizon |"
    )

    report.add_solution_method(
        "The computation uses plain grid VFI. For each current asset grid point, it "
        "evaluates every feasible next-asset choice, rules out choices with "
        "$Ra+y-a'\\le 0$, and keeps the maximizing value and policy. This is slow relative "
        "to endogenous-grid methods, but it makes the Bellman problem transparent.\n\n"
        "```text\n"
        "Input: asset grid A, primitives beta, R, y, gamma, tolerance epsilon\n"
        "Initialize V_0(a) = u(ra + y) / (1 - beta) for each a in A\n"
        "For n = 0, 1, 2, ...:\n"
        "    For each current asset a in A:\n"
        "        For each candidate next asset a' in A:\n"
        "            If c = R a + y - a' > 0, compute u(c) + beta V_n(a')\n"
        "            Otherwise mark the choice infeasible\n"
        "        Set V_{n+1}(a) to the largest feasible value\n"
        "        Store g(a), the next-asset choice attaining that value\n"
        "    Stop when max_a |V_{n+1}(a) - V_n(a)| < epsilon\n"
        "Output: value function V, savings policy g, consumption policy c(a)\n"
        "```\n\n"
        "After solving the Bellman equation, the policy is simulated from many initial "
        "asset positions. In this deterministic setting the simulation is not a Monte "
        "Carlo approximation to aggregate risk; it is a way to show the transition map "
        "implied by the policy.\n\n"
        f"The VFI loop converged in **{iteration} iterations** with sup-norm error "
        f"{Vdiff:.2e}. The numerical stationary asset is $a^{{*}}={a_ss:.4f}$, matching "
        f"the exact benchmark $\\underline a={exact_ss_a:.4f}$ up to grid error "
        f"({ss_error:.2e}). Stationary consumption is $c^{{*}}={c_ss:.4f}$."
    )

    report.add_results(
        f"The calibration is intentionally impatient: $\\beta R={beta_r:.4f}<1$. "
        "That one inequality is enough to organize the results. The value function "
        "is still smooth and concave on the grid, but the policy points toward asset "
        "decumulation rather than long-run wealth accumulation."
    )

    # --- Figure 1: Value Function ---
    fig1, ax1 = plt.subplots()
    ax1.plot(agrid, V, "b-", linewidth=2)
    ax1.set_xlabel("Assets $a$")
    ax1.set_ylabel("$V(a)$")
    ax1.set_title("Value Function")
    ax1.set_xlim(0, amax)
    report.add_figure("figures/value-function.png", "Value function V(a) over the asset grid", fig1,
        description="The value function rises with assets, but its curvature is the main object: "
        "wealth is most valuable close to the borrowing limit. The deterministic environment "
        "does not remove curvature from preferences; it removes the risk motive "
        "for holding a buffer stock.")


    # --- Figure 2: Consumption Policy ---
    fig2, ax2 = plt.subplots()
    ax2.plot(agrid, con, "b-", linewidth=2, label="$c^{*}(a)$")
    ax2.plot(agrid, R * agrid + y, "k:", linewidth=0.8, alpha=0.5, label="Cash-on-hand $Ra + y$")
    ax2.scatter([exact_ss_a], [exact_ss_c], color="r", s=30, zorder=3, label="Exact steady state")
    ax2.set_xlabel("Assets $a$")
    ax2.set_ylabel("Consumption $c$")
    ax2.set_title("Consumption Policy Function")
    ax2.set_xlim(0, amax)
    ax2.legend()
    report.add_figure("figures/consumption-policy.png", "Consumption policy function c(a)", fig2,
        description="The consumption policy is far below cash-on-hand for wealthy households: "
        "they do not consume all assets at once, but they do consume enough to reduce assets "
        "over time. At the borrowing limit, the household simply consumes current income.")


    # --- Figure 3: Savings Policy ---
    fig3, ax3 = plt.subplots()
    ax3.plot(agrid, sav - agrid, "b-", linewidth=2, label="$a' - a$")
    ax3.axhline(0, color="k", linewidth=0.5)
    ax3.scatter([exact_ss_a], [0], color="r", s=30, zorder=3, label="Exact steady state")
    ax3.set_xlabel("Assets $a$")
    ax3.set_ylabel("Net savings $a' - a$")
    ax3.set_title("Savings Policy Function")
    ax3.set_xlim(0, amax)
    ax3.legend()
    report.add_figure("figures/savings-policy.png", "Savings policy: net change in assets a'-a as a function of current assets", fig3,
        description="The net-saving policy makes the no-risk benchmark stark. The only fixed "
        "point is the borrowing limit, so positive initial assets are gradually spent down. "
        "In the later income-risk tutorial, the same kind of policy plot bends upward because "
        "risk creates a reason to hold buffer wealth.")


    # --- Figure 4: Simulated Asset Dynamics ---
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    for i in range(min(Nsim, 20)):  # Plot a subset to avoid clutter
        ax4.plot(range(Tsim), asim[i, :], linewidth=0.5, alpha=0.6)
    ax4.axhline(exact_ss_a, color="r", linewidth=1.5, linestyle="--", label=f"Exact $a^{{*}} = {exact_ss_a:.2f}$")
    ax4.set_xlabel("Period")
    ax4.set_ylabel("Assets $a_t$")
    ax4.set_title("Simulated Asset Dynamics (20 agents)")
    ax4.set_xlim(0, Tsim)
    ax4.set_ylim(bottom=0)
    ax4.legend()
    report.add_figure("figures/asset-dynamics.png", "Simulated asset paths converging to steady state", fig4,
        description="The simulated paths start from different asset positions but all converge "
        "to the same lower bound. The exercise is therefore a transition experiment, not a "
        "claim that deterministic savings can generate cross-sectional wealth inequality.")


    # --- Table: Policy Function at Selected Grid Points ---
    sample_idx = np.unique(np.r_[0, 1, 2, 5, 10, 25, 50, 100, 250, 500, na - 1])
    table_data = {
        "Assets a": [f"{agrid[i]:.4f}" for i in sample_idx],
        "Consumption c(a)": [f"{con[i]:.4f}" for i in sample_idx],
        "Next assets g(a)": [f"{sav[i]:.4f}" for i in sample_idx],
        "Net saving g(a)-a": [f"{(sav[i] - agrid[i]):.4f}" for i in sample_idx],
        "V(a)": [f"{V[i]:.4f}" for i in sample_idx],
    }
    df = pd.DataFrame(table_data)
    report.add_table("tables/policy-function.csv", "Selected Policy Values", df,
        description="The table reports low-asset grid points explicitly because the action near "
        "the borrowing limit is easy to miss on a wide plot. Once assets are positive, the "
        "policy keeps $g(a)<a$ and moves the household back toward $a^{*}=0$.")


    report.add_takeaway(
        "The deterministic model is the baseline to subtract from richer household problems. "
        "Here the borrowing constraint matters, but it does not create a wealth distribution: "
        "with $\\beta R<1$, every positive asset position is temporary and the exact stationary "
        "asset level is $a^{*}=0$.\n\n"
        "This is why the neighboring [IID income-risk VFI tutorial](../vfi-iid-income/) is not "
        "just the same code with an extra shock. Once income is risky, assets become self-insurance "
        "and the stationary distribution is an economic object. The later "
        "[endogenous-grid tutorial](../endogenous-grid-points/) keeps that economic problem but "
        "changes the solver."
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
