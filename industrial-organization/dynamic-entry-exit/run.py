#!/usr/bin/env python3
"""Dynamic Entry and Exit: Firm Turnover in Oligopolistic Markets.

Solves a dynamic discrete choice model of firm entry and exit in the spirit of
Rust (1987) and Ericson-Pakes (1995). Firms make binary stay/exit decisions each
period; potential entrants decide whether to pay a sunk cost to enter. The model
generates a stationary equilibrium distribution of market structure (number of
active firms) with simultaneous entry and exit -- "churning" -- even in steady state.

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


# =============================================================================
# Model primitives
# =============================================================================

def cournot_profit(N, a, b, c):
    """Per-firm Cournot profit with N symmetric firms.

    pi(N) = (a - c)^2 / (b * (N+1)^2)
    """
    return (a - c) ** 2 / (b * (N + 1) ** 2)


def solve_model(N_max, a, b, c, f, K, beta, sigma_eps=1.0, tol=1e-8, max_iter=5000):
    """Solve for the Markov-perfect equilibrium of the entry/exit game.

    Each period:
      1. N incumbents observe the state and draw idiosyncratic cost shocks
         epsilon_i ~ Logistic(0, sigma_eps). An incumbent stays iff:
             pi(N) - f + epsilon_i + beta * E[V(N')] >= 0
         This gives a smooth (logistic) exit probability at each N.
      2. Potential entrants enter until the free-entry condition binds:
         E[V(N_post)] <= K, where N_post includes survivors + new entrants.

    We iterate on the value function V(N) using dampened VFI. The value function
    here is the "pre-shock" expected value (integrating over the logistic shock):
        V(N) = sigma_eps * log(1 + exp((pi(N) - f + beta*EV(N)) / sigma_eps))
    This is the log-sum formula from the logit discrete choice model.
    """
    N_grid = np.arange(1, N_max + 1)
    n_states = N_max

    # Flow profits at each N
    profits = np.array([cournot_profit(N, a, b, c) for N in N_grid])

    # Initialize value function: myopic value
    V = np.maximum(profits - f, 0.0) / (1.0 - beta)

    # Dampening factor for stability
    dampen = 0.3

    for iteration in range(1, max_iter + 1):
        V_new = np.zeros(n_states)

        for i in range(n_states):
            N = N_grid[i]
            pi_N = profits[i]

            # --- Compute exit probability from current V ---
            # Continuation utility (net of shock) for an incumbent:
            #   u_stay(N) = pi(N) - f + beta * EV(N)
            # Exit probability (logistic shock):
            #   p_exit(N) = 1 / (1 + exp(u_stay / sigma_eps))
            # But EV(N) itself depends on the transition, which depends on p_exit.
            # We use the PREVIOUS iteration's V to compute EV, breaking the circularity.

            # For EV(N): integrate over survivors. Each of the other N-1 incumbents
            # stays with probability p_stay (from previous iteration). Then free
            # entry adds entrants. We condition on this firm staying.
            p_exit_i = _exit_prob(N, profits, f, beta, V, sigma_eps)
            p_stay_others = 1.0 - p_exit_i

            # E[V(N')] integrating over binomial survivors of the OTHER N-1 firms
            EV = 0.0
            for s in range(N):  # s = survivors among other N-1 firms
                prob_s = binom.pmf(s, N - 1, p_stay_others) if N > 1 else (1.0 if s == 0 else 0.0)
                if prob_s < 1e-15:
                    continue

                # This firm stays => N_survivors = s + 1
                N_surv = s + 1

                # Free entry: entrants enter until marginal entrant's value < K
                n_enter = _free_entry_count(N_surv, profits, f, beta, V, K, N_max, sigma_eps)
                N_next = min(N_surv + n_enter, N_max)

                EV += prob_s * V[N_next - 1]

            # Value of staying (deterministic component)
            u_stay = pi_N - f + beta * EV

            # Inclusive value (expected value integrating over logistic shock):
            # V(N) = sigma * log(exp(u_stay/sigma) + exp(0/sigma))
            #       = sigma * log(1 + exp(u_stay/sigma))
            # This is the "log-sum" formula from McFadden (1978)
            V_new[i] = sigma_eps * np.logaddexp(u_stay / sigma_eps, 0.0)

        # Dampened update
        V_update = dampen * V_new + (1.0 - dampen) * V
        error = np.max(np.abs(V_update - V))

        if iteration % 100 == 0:
            print(f"  VFI iteration {iteration:4d}, error = {error:.2e}")

        V = V_update

        if error < tol:
            print(f"  VFI converged in {iteration} iterations (error = {error:.2e})")
            break

    # --- Extract equilibrium policies ---
    exit_prob = np.zeros(n_states)
    entry_count = np.zeros(n_states)

    for i in range(n_states):
        N = N_grid[i]
        exit_prob[i] = _exit_prob(N, profits, f, beta, V, sigma_eps)

        # Average entry (using expected survivors)
        p_stay = 1.0 - exit_prob[i]
        expected_surv = max(1, int(np.round(N * p_stay)))
        entry_count[i] = _free_entry_count(expected_surv, profits, f, beta, V, K, N_max, sigma_eps)

    info = {"iterations": iteration, "converged": error < tol, "error": error}
    return V, exit_prob, entry_count, N_grid, info


def _exit_prob(N, profits, f, beta, V, sigma_eps):
    """Compute equilibrium exit probability at state N.

    With logistic idiosyncratic shocks, P(exit) = 1/(1 + exp(u_stay/sigma)).
    Here u_stay uses a rough E[V(N')] based on current V at the expected next state.
    """
    pi_N = profits[N - 1]

    # Rough continuation: assume N stays roughly the same (self-consistent approx)
    # This is used only for computing the exit probability
    EV_approx = V[N - 1]
    u_stay = pi_N - f + beta * EV_approx
    return 1.0 / (1.0 + np.exp(u_stay / sigma_eps))


def _free_entry_count(N_surv, profits, f, beta, V, K, N_max, sigma_eps):
    """Compute number of entrants given N_surv survivors.

    Entrants enter until the value of being in a market with (N_surv + n_enter)
    firms is less than the sunk cost K.
    """
    n_enter = 0
    while N_surv + n_enter < N_max:
        N_post = N_surv + n_enter + 1
        # An entrant would become an incumbent in a market with N_post firms
        # Their value is V(N_post) (the inclusive value)
        if V[N_post - 1] >= K:
            n_enter += 1
        else:
            break
    return n_enter


def compute_transition_matrix(N_max, exit_prob, entry_count):
    """Build the Markov transition matrix P[N, N'].

    At state N: each incumbent exits with prob exit_prob[N], survivors are binomial.
    Then entry_count[N] entrants arrive (deterministic, based on free entry).
    """
    P = np.zeros((N_max, N_max))

    for i in range(N_max):
        N = i + 1
        p_stay = 1.0 - exit_prob[i]
        n_enter = int(np.round(entry_count[i]))

        for s in range(N + 1):  # s = number of survivors out of N
            prob_s = binom.pmf(s, N, p_stay)
            if prob_s < 1e-15:
                continue

            N_next = s + n_enter
            N_next = max(1, min(N_next, N_max))
            P[i, N_next - 1] += prob_s

    # Normalize for numerical safety
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = P / row_sums

    return P


def compute_stationary_distribution(P, tol=1e-14, max_iter=50000):
    """Find stationary distribution pi such that pi = pi @ P."""
    n = P.shape[0]
    pi = np.ones(n) / n

    for it in range(max_iter):
        pi_new = pi @ P
        pi_new = pi_new / pi_new.sum()  # normalize
        if np.max(np.abs(pi_new - pi)) < tol:
            break
        pi = pi_new

    return pi_new


def simulate_market(T, N_init, exit_prob, entry_count, N_max, rng=None):
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

        # Each incumbent exits independently
        n_exits = rng.binomial(N, p_exit)
        survivors = N - n_exits

        # Entrants: Poisson noise around the expected entry count
        expected_enter = entry_count[idx]
        n_enter = rng.poisson(max(expected_enter, 0.0))

        N_next = max(1, min(survivors + n_enter, N_max))
        N_path[t + 1] = N_next
        exit_path[t] = n_exits
        entry_path[t] = n_enter

    return N_path, entry_path, exit_path


# =============================================================================
# Main
# =============================================================================

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
    sigma_eps = 1.0  # Scale of idiosyncratic logistic shock
    tol = 1e-8    # Convergence tolerance

    # =========================================================================
    # Solve the model
    # =========================================================================
    print("Solving dynamic entry/exit model...")
    V, exit_prob, entry_count, N_grid, info = solve_model(
        N_max, a, b, c, f, K, beta, sigma_eps=sigma_eps, tol=tol
    )

    # =========================================================================
    # Transition matrix and stationary distribution
    # =========================================================================
    print("Computing stationary distribution...")
    P = compute_transition_matrix(N_max, exit_prob, entry_count)
    stat_dist = compute_stationary_distribution(P)

    # =========================================================================
    # Simulate market evolution
    # =========================================================================
    print("Simulating market evolution...")
    T_sim = 200
    N_init = 5
    N_path, entry_path, exit_path = simulate_market(
        T_sim, N_init, exit_prob, entry_count, N_max
    )

    # =========================================================================
    # Equilibrium statistics
    # =========================================================================
    expected_N = np.sum(N_grid * stat_dist)
    std_N = np.sqrt(np.sum((N_grid - expected_N) ** 2 * stat_dist))
    mode_N = N_grid[np.argmax(stat_dist)]
    profits_at_mean = cournot_profit(int(np.round(expected_N)), a, b, c)
    expected_exit_rate = np.sum(exit_prob * stat_dist)
    expected_entry = np.sum(entry_count * stat_dist)

    # Per-firm profits
    profits = np.array([cournot_profit(N, a, b, c) for N in N_grid])

    # HHI at expected N (symmetric firms: HHI = 10000/N)
    hhi_at_mean = 10000.0 / max(1, int(np.round(expected_N)))

    # Zero-profit N: where pi(N) = f
    # (a-c)^2 / (b*(N+1)^2) = f  =>  N+1 = (a-c)/sqrt(b*f)  =>  N = (a-c)/sqrt(b*f) - 1
    N_zero_profit = (a - c) / np.sqrt(b * f) - 1

    print(f"\n  E[N] = {expected_N:.2f}, mode = {mode_N}, zero-profit N = {N_zero_profit:.1f}")

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
        "This captures a key empirical regularity in industrial organization -- markets exhibit "
        "substantial firm turnover despite relatively stable aggregate concentration."
    )

    report.add_equations(
        r"""
**Per-firm Cournot profit with $N$ symmetric firms:**

$$\pi(N) = \frac{(a - c)^2}{b \cdot (N+1)^2}$$

**Incumbent's value function (with logistic idiosyncratic shock $\varepsilon$):**

$$V_I(N) = \sigma_\varepsilon \cdot \log\!\left(1 + \exp\!\left(\frac{\pi(N) - f + \beta \, \mathbb{E}[V_I(N')]}{\sigma_\varepsilon}\right)\right)$$

This is the log-sum (inclusive value) from the logit model. An incumbent stays iff
$\pi(N) - f + \varepsilon + \beta \, \mathbb{E}[V(N')] \geq 0$; the logistic shock
generates a smooth exit probability:

$$p_{\text{exit}}(N) = \frac{1}{1 + \exp\!\big((\pi(N) - f + \beta \, \mathbb{E}[V(N')]) / \sigma_\varepsilon\big)}$$

**Free entry condition:**

$$V_I(N') \leq K$$

Potential entrants enter until the expected value of incumbency (in the post-entry market)
falls below the sunk cost $K$.

**Transition:**

$$N' = \text{Binomial survivors}(N, 1 - p_{\text{exit}}) + \text{Entrants}$$
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
        f"| $\\sigma_\\varepsilon$ | {sigma_eps} | Logistic shock scale |\n"
        f"| $N_{{\\max}}$ | {N_max} | Maximum number of firms |"
    )

    report.add_solution_method(
        "**Dampened Value Function Iteration (VFI)** with log-sum inclusive values:\n\n"
        "1. Initialize $V(N)$ for all states $N = 1, \\ldots, N_{\\max}$.\n"
        "2. For each state, compute the exit probability from the logistic choice model "
        "using the current $V$.\n"
        "3. Compute $\\mathbb{E}[V(N')]$ by integrating over the binomial distribution "
        "of survivors (other $N-1$ incumbents), with free entry determining entrants "
        "at each realization.\n"
        "4. Update $V(N)$ using the log-sum formula with dampening factor 0.3.\n"
        "5. Iterate until $\\|V_{n+1} - V_n\\|_\\infty < 10^{-8}$.\n\n"
        f"Converged in **{info['iterations']} iterations** (error = {info['error']:.2e}).\n\n"
        "The stationary distribution is computed by constructing the Markov transition "
        "matrix $P(N' | N)$ and finding its invariant distribution via power iteration."
    )

    # --- Figure 1: Value Function ---
    fig1, ax1 = plt.subplots()
    ax1.plot(N_grid, V, "b-o", markersize=4, linewidth=2, label="$V(N)$")
    ax1.axhline(y=0, color="k", linewidth=0.5, linestyle="--")
    ax1.axhline(y=K, color="r", linewidth=1, linestyle="--", alpha=0.7, label=f"Sunk cost $K = {K}$")
    ax1.axvline(x=N_zero_profit, color="gray", linewidth=1, linestyle=":", alpha=0.7,
                label=f"Zero-profit $N = {N_zero_profit:.1f}$")
    ax1.set_xlabel("Number of firms $N$")
    ax1.set_ylabel("Value $V(N)$")
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
    ax2a.set_ylim(bottom=0)

    ax2b = ax2a.twinx()
    ax2b.plot(N_grid, entry_count, "s-", color=color_entry, markersize=4, linewidth=2, label="Expected entrants")
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
    ax3.axvline(x=expected_N, color="red", linewidth=1.5, linestyle="--",
                label=f"$E[N] = {expected_N:.1f}$")
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
    ax4a.axhline(y=expected_N, color="red", linewidth=1, linestyle="--", alpha=0.7,
                 label=f"$E[N] = {expected_N:.1f}$")
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

    # --- Table 1: Equilibrium Statistics ---
    stats_data = {
        "Statistic": [
            "Expected number of firms E[N]",
            "Std. deviation of N",
            "Modal number of firms",
            "Zero-profit N (static)",
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
            f"{N_zero_profit:.1f}",
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

    # --- Table 2: Value and Policies by N ---
    sample_N = np.array([1, 2, 3, 5, 7, 10, 15, 20, 25, 30])
    sample_N = sample_N[sample_N <= N_max]
    detail_data = {
        "N": [str(n) for n in sample_N],
        "Profit pi(N)": [f"{cournot_profit(n, a, b, c):.3f}" for n in sample_N],
        "Net profit pi-f": [f"{cournot_profit(n, a, b, c) - f:.3f}" for n in sample_N],
        "V(N)": [f"{V[n - 1]:.3f}" for n in sample_N],
        "Exit prob": [f"{exit_prob[n - 1]:.4f}" for n in sample_N],
        "Entry": [f"{entry_count[n - 1]:.0f}" for n in sample_N],
    }
    df_detail = pd.DataFrame(detail_data)
    report.add_table(
        "tables/value-by-N.csv",
        "Value Function and Policies at Selected Market Structures",
        df_detail,
    )

    report.add_takeaway(
        "Dynamic entry/exit models explain why markets have persistent differences in "
        "concentration. Entry costs create barriers that sustain above-competitive profits, "
        "while exit occurs when negative shocks or increased competition erode incumbents' "
        "continuation values.\n\n"
        "**Key insights:**\n"
        "- The value of incumbency declines sharply with $N$: more competitors erode Cournot "
        "rents. Beyond a threshold, $V(N) \\approx 0$ and firms prefer to exit.\n"
        "- The sunk cost $K$ creates hysteresis: incumbents only face the per-period cost $f$ "
        "to stay, while entrants must pay $K$ up front. This wedge between entry and exit "
        "thresholds is the source of persistence in market structure.\n"
        "- The model generates \"churning\" -- simultaneous entry and exit even in steady state -- "
        "because idiosyncratic shocks push some incumbents below the exit threshold while "
        "the market remains attractive enough for new entrants.\n"
        "- The stationary distribution concentrates near the free-entry equilibrium $N$, but "
        "stochastic turnover generates a non-degenerate spread around this point."
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
