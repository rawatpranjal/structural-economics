#!/usr/bin/env python3
"""Dynamic Entry and Exit: Firm Turnover in Oligopolistic Markets.

Solves a dynamic discrete choice model of firm entry and exit in the spirit of
Rust (1987) and Ericson-Pakes (1995). Firms make binary stay/exit decisions each
period; potential entrants decide whether to pay a sunk cost to enter. The model
generates a stationary equilibrium distribution of market structure (number of
active firms) with simultaneous entry and exit — "churning" — even in steady state.

Reference: Ericson and Pakes (1995), Hopenhayn (1992).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binom

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def cournot_profit(N, a, b, c):
    """Per-firm Cournot profit with N symmetric firms.

    pi(N) = (a - c)^2 / (b * (N+1)^2)
    """
    return (a - c) ** 2 / (b * (N + 1) ** 2)


def solve_value_function(N_max, a, b, c, f, K, beta, tol=1e-8, max_iter=2000):
    """Solve the incumbent's value function via VFI.

    The state is the number of active firms N. Each incumbent observes N and
    decides whether to stay (earning pi(N) - f + beta * E[V(N')]) or exit (0).

    Transition: N' = survivors + entrants, where each of the other (N-1)
    incumbents survives with the equilibrium probability, and entry is
    determined by a free-entry condition.

    We iterate jointly on V and the exit/entry probabilities until convergence.
    """
    N_grid = np.arange(1, N_max + 1)  # N = 1, ..., N_max
    n_states = len(N_grid)

    # Initialize value function
    V = np.zeros(n_states)
    for i, N in enumerate(N_grid):
        V[i] = max(cournot_profit(N, a, b, c) - f, 0.0) / (1 - beta)

    # Storage for policies
    exit_prob = np.zeros(n_states)     # probability an incumbent exits at state N
    entry_rate = np.zeros(n_states)    # expected number of entrants at state N

    for iteration in range(1, max_iter + 1):
        V_new = np.zeros(n_states)
        exit_prob_new = np.zeros(n_states)
        entry_rate_new = np.zeros(n_states)

        for i, N in enumerate(N_grid):
            # --- Flow profit ---
            pi_N = cournot_profit(N, a, b, c)

            # --- Exit decision ---
            # Each incumbent compares continuation value to zero.
            # The continuation value depends on the expected future state.
            # For tractability, we compute E[V(N')] given current policies,
            # then check whether staying is optimal.

            # Expected value of staying given current V (using current state as
            # rough proxy for expected future state):
            # We need E[V(N')]. N' depends on how many of the OTHER incumbents
            # stay and how many entrants arrive.

            # Step 1: Compute exit probability from indifference condition.
            # In a symmetric MPE, if V_stay > 0, all incumbents stay; if < 0, exit.
            # With heterogeneous exit costs (or idiosyncratic shocks), we get
            # interior exit probabilities. We model this with a logistic smoothing
            # to get an interior equilibrium.

            # Expected continuation value (weighted average over possible N'):
            EV = 0.0
            # The other N-1 incumbents each stay with prob (1 - exit_prob[i])
            p_stay_others = 1.0 - exit_prob[i]

            # Expected survivors (besides this firm): Binomial(N-1, p_stay_others)
            # Plus this firm (if it stays) = survivors + 1
            # Plus entrants

            # Compute E[V] by averaging over possible survivor counts
            for s in range(N):
                # s = number of other incumbents who survive (0 to N-1)
                prob_s = binom.pmf(s, N - 1, p_stay_others)
                if prob_s < 1e-12:
                    continue

                # If this firm stays, next period has (s + 1 + entrants) firms
                N_survivors = s + 1  # including this firm

                # Free entry: entrants enter until E[V(N' + 1)] <= K
                # Determine number of entrants given survivors
                n_enter = 0
                while N_survivors + n_enter + 1 <= N_max:
                    N_test = N_survivors + n_enter + 1
                    idx_test = N_test - 1  # index into V
                    if idx_test < n_states:
                        entry_value = pi_N_at(N_test, a, b, c) - f + beta * V[idx_test]
                        if entry_value >= K:
                            n_enter += 1
                        else:
                            break
                    else:
                        break

                N_next = min(N_survivors + n_enter, N_max)
                idx_next = N_next - 1
                EV += prob_s * V[idx_next]

            # Value of staying
            V_stay = pi_N - f + beta * EV

            # Exit decision: stay if V_stay > 0 (with smooth logistic for stability)
            # Use a "smoothed" exit probability: sigma(-V_stay / temperature)
            temperature = 0.1
            exit_prob_new[i] = 1.0 / (1.0 + np.exp(V_stay / temperature))

            V_new[i] = max(V_stay, 0.0)

            # Record entry rate for this state
            entry_rate_new[i] = compute_entry_rate(N, exit_prob_new[i], V, a, b, c, f, K, beta, N_max)

        # Check convergence
        error = np.max(np.abs(V_new - V))
        if iteration % 50 == 0:
            print(f"  VFI iteration {iteration:4d}, error = {error:.2e}")

        V = V_new
        exit_prob = exit_prob_new
        entry_rate = entry_rate_new

        if error < tol:
            print(f"  VFI converged in {iteration} iterations (error = {error:.2e})")
            break

    return V, exit_prob, entry_rate, N_grid, {"iterations": iteration, "converged": error < tol, "error": error}


def pi_N_at(N, a, b, c):
    """Cournot profit helper for integer N."""
    return (a - c) ** 2 / (b * (N + 1) ** 2)


def compute_entry_rate(N, exit_p, V, a, b, c, f, K, beta, N_max):
    """Compute expected number of entrants given state N and exit probability.

    Free entry: firms enter until the expected value of entry (post-entry V minus
    sunk cost K) is non-positive.
    """
    # Expected survivors
    p_stay = 1.0 - exit_p
    expected_survivors = max(1, int(np.round(N * p_stay)))

    n_enter = 0
    while expected_survivors + n_enter + 1 <= N_max:
        N_post = expected_survivors + n_enter + 1
        idx = N_post - 1
        if idx < len(V):
            # An entrant becomes an incumbent next period in a market with N_post firms
            entry_value = cournot_profit(N_post, a, b, c) - f + beta * V[idx] - K
            if entry_value >= 0:
                n_enter += 1
            else:
                break
        else:
            break

    return n_enter


def compute_stationary_distribution(N_max, exit_prob, entry_rate, a, b, c, f, K, beta):
    """Compute the stationary distribution of N via transition matrix iteration.

    Build the Markov transition matrix P[N, N'] and find its stationary distribution.
    """
    n_states = N_max
    P = np.zeros((n_states, n_states))

    for i in range(n_states):
        N = i + 1  # current number of firms
        p_stay = 1.0 - exit_prob[i]
        n_enter = int(np.round(entry_rate[i]))

        # Transition: each of N incumbents stays with prob p_stay (binomial)
        for s in range(N + 1):
            prob_s = binom.pmf(s, N, p_stay)
            if prob_s < 1e-12:
                continue

            N_next = s + n_enter
            N_next = max(1, min(N_next, N_max))  # clamp to [1, N_max]
            j = N_next - 1
            P[i, j] += prob_s

    # Normalize rows (should already sum to 1, but ensure numerical stability)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = P / row_sums

    # Find stationary distribution by iterating pi = pi @ P
    pi = np.ones(n_states) / n_states
    for _ in range(10000):
        pi_new = pi @ P
        if np.max(np.abs(pi_new - pi)) < 1e-12:
            break
        pi = pi_new

    return pi, P


def simulate_market(T, N_init, exit_prob, entry_rate, N_max, rng=None):
    """Simulate market evolution for T periods."""
    if rng is None:
        rng = np.random.default_rng(42)

    N_path = np.zeros(T, dtype=int)
    N_path[0] = N_init
    entry_path = np.zeros(T, dtype=int)
    exit_path = np.zeros(T, dtype=int)

    for t in range(T - 1):
        N = N_path[t]
        idx = min(N - 1, len(exit_prob) - 1)
        p_exit = exit_prob[idx]

        # Each incumbent exits independently with probability p_exit
        n_exits = rng.binomial(N, p_exit)
        survivors = N - n_exits

        # Entrants
        n_enter = int(np.round(entry_rate[idx]))
        # Add some randomness to entry (Poisson around expected)
        n_enter = rng.poisson(max(n_enter, 0))

        N_next = max(1, min(survivors + n_enter, N_max))
        N_path[t + 1] = N_next
        exit_path[t] = n_exits
        entry_path[t] = n_enter

    return N_path, entry_path, exit_path


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    a = 10        # Demand intercept
    b = 1         # Demand slope
    c = 2         # Marginal cost
    f = 0.5       # Fixed cost (per period)
    K = 5.0       # Sunk entry cost
    beta = 0.95   # Discount factor
    N_max = 30    # Maximum number of firms
    tol = 1e-8    # Convergence tolerance

    # =========================================================================
    # Solve the model
    # =========================================================================
    print("Solving dynamic entry/exit model...")
    V, exit_prob, entry_rate, N_grid, info = solve_value_function(
        N_max, a, b, c, f, K, beta, tol=tol
    )

    # =========================================================================
    # Compute stationary distribution
    # =========================================================================
    print("Computing stationary distribution...")
    stat_dist, P = compute_stationary_distribution(
        N_max, exit_prob, entry_rate, a, b, c, f, K, beta
    )

    # =========================================================================
    # Simulate market evolution
    # =========================================================================
    print("Simulating market evolution...")
    T_sim = 200
    N_init = 5
    N_path, entry_path, exit_path = simulate_market(
        T_sim, N_init, exit_prob, entry_rate, N_max
    )

    # =========================================================================
    # Compute equilibrium statistics
    # =========================================================================
    expected_N = np.sum(N_grid * stat_dist)
    std_N = np.sqrt(np.sum((N_grid - expected_N) ** 2 * stat_dist))
    mode_N = N_grid[np.argmax(stat_dist)]
    profits_at_mean = cournot_profit(int(np.round(expected_N)), a, b, c)
    expected_exit_rate = np.sum(exit_prob * stat_dist)
    expected_entry = np.sum(entry_rate * stat_dist)

    # Per-firm profit at each N
    profits = np.array([cournot_profit(N, a, b, c) for N in N_grid])
    net_profits = profits - f

    # HHI at expected N
    hhi_at_mean = 10000 / int(np.round(expected_N))  # Symmetric firms

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Dynamic Entry and Exit",
        "Firm turnover and market structure in an oligopolistic industry with sunk entry costs.",
    )

    report.add_overview(
        "This model studies how firms' entry and exit decisions determine market structure "
        "over time. Each period, incumbent firms decide whether to continue operating (paying "
        "a fixed cost $f$) or exit permanently. Simultaneously, potential entrants decide "
        "whether to pay a sunk cost $K$ to enter the market. Firms compete as Cournot "
        "oligopolists, so profits depend on the number of active firms.\n\n"
        "The model generates a stationary equilibrium with persistent heterogeneity in market "
        "structure: even in steady state, there is simultaneous entry and exit (\"churning\"). "
        "This captures a key empirical regularity in industrial organization — markets exhibit "
        "substantial firm turnover despite relatively stable aggregate concentration."
    )

    report.add_equations(
        r"""
**Per-firm Cournot profit with $N$ symmetric firms:**

$$\pi(N) = \frac{(a - c)^2}{b \cdot (N+1)^2}$$

**Incumbent's value function:**

$$V_I(N) = \max\left\{ \pi(N) - f + \beta \, \mathbb{E}[V_I(N')], \quad 0 \right\}$$

The first term is the value of staying (flow profit minus fixed cost, plus discounted
continuation value). The second term (zero) is the value of exiting.

**Free entry condition:**

$$\mathbb{E}[V_I(N')] \leq K$$

with equality if entry is positive. Potential entrants enter until the expected value
of being an incumbent (post-entry) equals the sunk cost $K$.

**Transition:**

$$N' = \text{Survivors}(N, p_{\text{exit}}) + \text{Entrants}(N)$$

where survivors follow a Binomial distribution and entry is determined by free entry.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $a$       | {a}  | Demand intercept |\n"
        f"| $b$       | {b}  | Demand slope |\n"
        f"| $c$       | {c}  | Marginal cost |\n"
        f"| $f$       | {f}  | Fixed operating cost (per period) |\n"
        f"| $K$       | {K}  | Sunk entry cost |\n"
        f"| $\\beta$  | {beta} | Discount factor |\n"
        f"| $N_{{\\max}}$ | {N_max} | Maximum number of firms |"
    )

    report.add_solution_method(
        "**Value Function Iteration (VFI)** with simultaneous computation of exit and "
        "entry policies:\n\n"
        "1. Initialize $V(N)$ for all states $N = 1, \\ldots, N_{\\max}$.\n"
        "2. For each state $N$, compute the continuation value by integrating over "
        "possible transitions (binomial survival of other incumbents).\n"
        "3. Determine exit probability via a smoothed (logistic) best response: "
        "firms exit when $V_{\\text{stay}} < 0$.\n"
        "4. Determine entry via free entry: entrants enter until the marginal entrant's "
        "value falls below $K$.\n"
        "5. Iterate until $\\|V_{n+1} - V_n\\|_\\infty < 10^{-8}$.\n\n"
        f"Converged in **{info['iterations']} iterations** (error = {info['error']:.2e}).\n\n"
        "The stationary distribution is computed by constructing the Markov transition "
        "matrix $P(N' | N)$ and finding its invariant distribution via power iteration."
    )

    # --- Figure 1: Value Function ---
    fig1, ax1 = plt.subplots()
    ax1.plot(N_grid, V, "b-o", markersize=4, linewidth=2, label="$V_I(N)$")
    ax1.axhline(y=0, color="k", linewidth=0.5, linestyle="--")
    ax1.axhline(y=K, color="r", linewidth=1, linestyle="--", alpha=0.7, label=f"Sunk cost $K = {K}$")
    ax1.set_xlabel("Number of firms $N$")
    ax1.set_ylabel("Value $V_I(N)$")
    ax1.set_title("Incumbent Value Function")
    ax1.legend()
    report.add_figure(
        "figures/value-function.png",
        "Incumbent value function V(N): value of being an active firm as a function of market structure",
        fig1,
    )

    # --- Figure 2: Entry and Exit Probabilities ---
    fig2, ax2a = plt.subplots()
    color_exit = "tab:red"
    color_entry = "tab:blue"

    ax2a.plot(N_grid, exit_prob, "o-", color=color_exit, markersize=4, linewidth=2, label="Exit probability")
    ax2a.set_xlabel("Number of firms $N$")
    ax2a.set_ylabel("Exit probability", color=color_exit)
    ax2a.tick_params(axis="y", labelcolor=color_exit)

    ax2b = ax2a.twinx()
    ax2b.plot(N_grid, entry_rate, "s-", color=color_entry, markersize=4, linewidth=2, label="Expected entrants")
    ax2b.set_ylabel("Expected entrants", color=color_entry)
    ax2b.tick_params(axis="y", labelcolor=color_entry)

    ax2a.set_title("Entry and Exit vs Market Structure")
    lines1, labels1 = ax2a.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2a.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    report.add_figure(
        "figures/entry-exit-probabilities.png",
        "Exit probability and expected entry as functions of the number of active firms",
        fig2,
    )

    # --- Figure 3: Stationary Distribution ---
    fig3, ax3 = plt.subplots()
    ax3.bar(N_grid, stat_dist, color="steelblue", alpha=0.8, edgecolor="navy", linewidth=0.5)
    ax3.axvline(x=expected_N, color="red", linewidth=1.5, linestyle="--", label=f"$E[N] = {expected_N:.1f}$")
    ax3.set_xlabel("Number of firms $N$")
    ax3.set_ylabel("Probability")
    ax3.set_title("Stationary Distribution of Market Structure")
    ax3.legend()
    report.add_figure(
        "figures/stationary-distribution.png",
        "Stationary distribution of the number of active firms",
        fig3,
    )

    # --- Figure 4: Simulated Market Evolution ---
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    periods = np.arange(T_sim)

    ax4a.plot(periods, N_path, "b-", linewidth=1.2, alpha=0.8)
    ax4a.axhline(y=expected_N, color="red", linewidth=1, linestyle="--", alpha=0.7, label=f"$E[N] = {expected_N:.1f}$")
    ax4a.set_ylabel("Number of firms $N_t$")
    ax4a.set_title("Simulated Market Evolution")
    ax4a.legend()

    ax4b.bar(periods[:-1], entry_path[:-1], color="steelblue", alpha=0.6, label="Entries", width=1.0)
    ax4b.bar(periods[:-1], -exit_path[:-1], color="firebrick", alpha=0.6, label="Exits", width=1.0)
    ax4b.axhline(y=0, color="k", linewidth=0.5)
    ax4b.set_xlabel("Period $t$")
    ax4b.set_ylabel("Firms entering / exiting")
    ax4b.set_title("Entry and Exit Over Time")
    ax4b.legend()

    fig4.tight_layout()
    report.add_figure(
        "figures/simulated-market.png",
        "Simulated market: number of firms and entry/exit flows over 200 periods",
        fig4,
    )

    # --- Table: Equilibrium Statistics ---
    stats_data = {
        "Statistic": [
            "Expected number of firms E[N]",
            "Std. deviation of N",
            "Modal number of firms",
            "Per-firm profit at E[N]",
            "Net profit (pi - f) at E[N]",
            "HHI at E[N]",
            "Expected exit rate",
            "Expected entry (firms/period)",
            "VFI iterations",
        ],
        "Value": [
            f"{expected_N:.2f}",
            f"{std_N:.2f}",
            f"{mode_N}",
            f"{profits_at_mean:.3f}",
            f"{profits_at_mean - f:.3f}",
            f"{hhi_at_mean:.0f}",
            f"{expected_exit_rate:.4f}",
            f"{expected_entry:.2f}",
            f"{info['iterations']}",
        ],
    }
    df_stats = pd.DataFrame(stats_data)
    report.add_table("tables/equilibrium-statistics.csv", "Equilibrium Statistics", df_stats)

    # --- Table: Profit and Value by N ---
    sample_N = np.array([1, 2, 3, 5, 7, 10, 15, 20, 25, 30])
    sample_N = sample_N[sample_N <= N_max]
    detail_data = {
        "N": [str(n) for n in sample_N],
        "Profit pi(N)": [f"{cournot_profit(n, a, b, c):.3f}" for n in sample_N],
        "Net profit pi-f": [f"{cournot_profit(n, a, b, c) - f:.3f}" for n in sample_N],
        "V(N)": [f"{V[n-1]:.3f}" for n in sample_N],
        "Exit prob": [f"{exit_prob[n-1]:.4f}" for n in sample_N],
        "Entry rate": [f"{entry_rate[n-1]:.1f}" for n in sample_N],
    }
    df_detail = pd.DataFrame(detail_data)
    report.add_table("tables/value-by-N.csv", "Value Function and Policies at Selected Market Structures", df_detail)

    report.add_takeaway(
        "Dynamic entry/exit models explain why markets have persistent differences in "
        "concentration. Entry costs create barriers that sustain above-competitive profits, "
        "while exit occurs when negative shocks or increased competition erode incumbents' "
        "continuation values.\n\n"
        "**Key insights:**\n"
        "- The value of incumbency declines sharply with $N$: more competitors erode Cournot "
        "rents. Beyond a threshold, $V(N) = 0$ and all firms prefer to exit.\n"
        "- The sunk cost $K$ creates hysteresis: firms that are already in the market stay "
        "(since they only face $f$), while potential entrants need $V > K$ to justify entry. "
        "This wedge between entry and exit thresholds is the source of inertia in market "
        "structure.\n"
        "- The model generates \"churning\" — simultaneous entry and exit even in steady state — "
        "because stochastic transitions create states where some firms find it unprofitable to "
        "continue while others find it attractive to enter.\n"
        "- The stationary distribution shows the long-run probability of each market structure. "
        "Markets spend most of their time near the modal $N$, but occasionally visit very "
        "concentrated or very competitive states."
    )

    report.add_references([
        "Ericson, R. and Pakes, A. (1995). Markov-perfect industry dynamics: A framework for empirical work. *Review of Economic Studies*, 62(1):53-82.",
        "Hopenhayn, H. (1992). Entry, exit, and firm dynamics in long run equilibrium. *Econometrica*, 60(5):1127-1150.",
        "Rust, J. (1987). Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher. *Econometrica*, 55(5):999-1033.",
        "Pakes, A. and McGuire, P. (1994). Computing Markov-perfect Nash equilibria: Numerical implications of a dynamic differentiated product model. *RAND Journal of Economics*, 25(4):555-589.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
