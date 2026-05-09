#!/usr/bin/env python3
"""Entry, exit, and market structure in an oligopoly.

Solves a small finite-state entry/exit model in the spirit of Hopenhayn (1992)
and Ericson-Pakes (1995). Incumbents decide whether to remain active, potential
entrants pay a sunk cost when continuation value is high enough, and the induced
Markov chain describes the long-run distribution of market structure.

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
    """Solve a symmetric Markov entry/exit approximation.

    Each period:
      1. N incumbents observe the state and draw idiosyncratic cost shocks
         epsilon_i ~ Logistic(0, sigma_eps). An incumbent stays iff:
             pi(N) - f + epsilon_i + beta * E[V(N')] >= 0
         This gives a smooth (logistic) exit probability at each N.
      2. Potential entrants choose entry from the current state, using expected
         survivor counts, until the post-entry value falls below K.

    We iterate on the value function V(N) using dampened VFI. The value function
    is the pre-shock expected value, integrating over the logistic shock:
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

            # Entry is chosen at the market-state level before the realized exit
            # draws are known. Entrants respond to expected survival, while the
            # realized transition still integrates over binomial incumbent exits.
            expected_surv_all = max(0, int(np.round(N * (1.0 - p_exit_i))))
            n_enter_current = _free_entry_count(
                expected_surv_all, profits, f, beta, V, K, N_max, sigma_eps
            )

            # E[V(N')] integrating over binomial survivors of the OTHER N-1 firms.
            EV = 0.0
            for s in range(N):  # s = survivors among other N-1 firms
                prob_s = binom.pmf(s, N - 1, p_stay_others) if N > 1 else (1.0 if s == 0 else 0.0)
                if prob_s < 1e-15:
                    continue

                # This firm stays => N_survivors = s + 1
                N_surv = s + 1
                N_next = min(N_surv + n_enter_current, N_max)

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
        p_stay = 1.0 - exit_prob[i]
        expected_surv = max(0, int(np.round(N * p_stay)))
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

    At state N, entry is fixed by the current state and expected survival.
    Realized incumbent survival then generates the next market size.
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

        # Entry is decided from the current market state before realized exits.
        n_enter = int(np.round(entry_count[idx]))

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
    profits = np.array([cournot_profit(N, a, b, c) for N in N_grid])

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
    T_check = 50_000
    N_check, _, _ = simulate_market(
        T_check,
        N_init,
        exit_prob,
        entry_count,
        N_max,
        rng=np.random.default_rng(123),
    )
    burn_in = 1_000
    sim_dist = np.bincount(N_check[burn_in:] - 1, minlength=N_max) / (T_check - burn_in)

    # =========================================================================
    # Equilibrium statistics
    # =========================================================================
    expected_N = np.sum(N_grid * stat_dist)
    std_N = np.sqrt(np.sum((N_grid - expected_N) ** 2 * stat_dist))
    mode_N = N_grid[np.argmax(stat_dist)]
    profits_at_mean = cournot_profit(int(np.round(expected_N)), a, b, c)
    expected_exit_rate = np.sum(exit_prob * stat_dist)
    expected_exits = np.sum(N_grid * exit_prob * stat_dist)
    expected_entry = np.sum(entry_count * stat_dist)
    max_dist_gap = np.max(np.abs(sim_dist - stat_dist))

    # Zero-profit N: where pi(N) = f
    # (a-c)^2 / (b*(N+1)^2) = f  =>  N+1 = (a-c)/sqrt(b*f)  =>  N = (a-c)/sqrt(b*f) - 1
    N_zero_profit = (a - c) / np.sqrt(b * f) - 1

    print(f"\n  E[N] = {expected_N:.2f}, mode = {mode_N}, zero-profit N = {N_zero_profit:.1f}")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Entry, Exit, and Market Structure in Oligopoly",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Consider a local market with eight active firms. Some firms may be close to leaving. "
        "Potential entrants wait outside because entry requires a sunk cost. The firm count is "
        "therefore a state variable.\n\n"
        "The model uses a symmetric Cournot market. The state is the active firm count $N_t$. "
        "Incumbents pay fixed cost $f$ to operate. Entrants pay sunk cost $K$ before earning "
        "future profits.\n\n"
        "Exit and entry rules depend on future market sizes. We solve a finite-state Bellman "
        "fixed point for incumbent values. The implied Markov chain gives persistence and the "
        "long-run firm-count distribution."
    )

    report.add_equations(
        r"""
Let $N_t\in\{1,\ldots,N_{\max}\}$ denote the number of active firms at the start
of period $t$. With inverse demand $P=a-bQ$ and constant marginal cost $c$, the
symmetric Cournot flow profit before fixed cost is

$$
\pi(N)=\frac{(a-c)^2}{b(N+1)^2}.
$$

An incumbent's exit value is normalized to zero. If it stays, its deterministic
surplus is

$$
\Delta(N)=\pi(N)-f+\beta \mathbb{E}\left[V(N_{t+1})\mid N_t=N,\text{ stay}\right].
$$

The idiosyncratic stay shock has logistic scale $\sigma_\varepsilon$. The
pre-shock value is the log-sum inclusive value

$$
V(N)=\sigma_\varepsilon
\log\left[1+\exp\left(\frac{\Delta(N)}{\sigma_\varepsilon}\right)\right],
$$

The incumbent exit probability is

$$
p_{\mathrm{exit}}(N)=
\frac{1}{1+\exp\{\Delta(N)/\sigma_\varepsilon\}}.
$$

Entry is decided at the current market state, before the realized exit draws.
Potential entrants use the expected survivor count
$\bar S(N_t)=\mathrm{round}\{N_t[1-p_{\mathrm{exit}}(N_t)]\}$ and enter until
the next entrant would not cover the sunk cost. Entrant $m$ enters only if its
post-entry value $V(\bar S(N_t)+m)$ is at least $K$. Thus

$$
e(N_t)=\max\{e\geq 0: \bar S(N_t)+e\leq N_{\max}
\ \text{and}\ V(\bar S(N_t)+m)\geq K\ \text{for all}\ m=1,\ldots,e\},
$$

with $e(N_t)=0$ when the first entrant does not cover $K$.

Survival and entry define the transition law.

$$
S_t\sim \mathrm{Binomial}\left(N_t,1-p_{\mathrm{exit}}(N_t)\right),
\qquad
N_{t+1}=\max\{1,\min(S_t+e(N_t),N_{\max})\}.
$$
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
        f"| $N_{{\\max}}$ | {N_max} | Maximum number of firms |\n"
        f"| State space | $1,\\ldots,{N_max}$ | Operating markets; zero-firm market entry is not modeled |\n"
        f"| Simulation periods | {T_sim} | Market path shown in the results |"
    )

    report.add_solution_method(
        "The numerical object is a fixed point in incumbent continuation values. Given $V(N)$, "
        "the exit rule, entry cutoff, and transition matrix follow. The fixed point prices "
        "incumbency as an option.\n\n"
        "The algorithm iterates on $V(N)$. Each pass computes exit probabilities and the "
        "free-entry cutoff. It then integrates over survivor counts and updates the log-sum "
        "value. The final policies define a Markov chain over firm counts.\n\n"
        "```text\n"
        "Algorithm: symmetric entry-exit fixed point\n"
        "Input: state grid {1,...,N_max}, primitives (a,b,c,f,K,beta,sigma), tolerance epsilon\n"
        "Output: V(N), p_exit(N), expected entry, transition matrix P, stationary distribution mu\n"
        "Initialize V_0(N) from myopic operating values\n"
        "repeat for n = 0, 1, 2, ...:\n"
        "    for each market size N:\n"
        "        compute current Cournot profit pi(N)\n"
        "        compute p_exit(N) from the logit stay/exit rule using V_n\n"
        "        compute expected survivor count S_bar(N)\n"
        "        choose entrants e(N) by the cutoff V_n(S_bar(N)+e) >= K\n"
        "        for each possible number of rival survivors S:\n"
        "            add V_n(min{S + 1 + e(N), N_max}) to the incumbent's continuation value\n"
        "        update V_{n+1}(N) with the log-sum inclusive value\n"
        "    replace V_n by a damped average of V_n and V_{n+1}\n"
        "until max_N |V_{n+1}(N)-V_n(N)| < epsilon\n"
        "Construct P(N'|N) from binomial survival and the same state-level entry rule\n"
        "Iterate mu_{m+1}=mu_m P until mu is invariant\n"
        "```\n\n"
        f"The value iteration converged in **{info['iterations']} iterations** with "
        f"sup-norm error **{info['error']:.2e}**. The invariant distribution solves "
        "$\\mu=\\mu P$ for the policy-induced Markov chain."
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
        "Incumbent value function by number of active firms",
        fig1,
        description=(
            "Incumbency value falls as more firms divide Cournot rents. The dashed line is the "
            "sunk entry cost. Below it, a new firm would not enter. An incumbent may still stay "
            "because it already paid the cost. The vertical line marks the static "
            "zero-flow-profit benchmark."
        ),
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
        "Exit probability and expected entry by market size",
        fig2,
        description=(
            "Exit risk rises with crowding because profits and continuation values fall. Expected "
            "entry is high when the market is thin. It falls once post-entry value drops below "
            "$K$. The gap between thresholds is the hysteresis region created by sunk entry."
        ),
    )

    # --- Figure 3: Stationary Distribution ---
    fig3, ax3 = plt.subplots()
    ax3.bar(N_grid, stat_dist, color="steelblue", alpha=0.8, edgecolor="navy", linewidth=0.5)
    ax3.plot(N_grid, sim_dist, "ko", markersize=3.5, label="Long simulation")
    ax3.axvline(x=expected_N, color="red", linewidth=1.5, linestyle="--",
                label=f"$E[N] = {expected_N:.1f}$")
    ax3.set_xlabel("Number of firms $N$")
    ax3.set_ylabel("Probability")
    ax3.set_title("Stationary Distribution of Market Structure")
    ax3.legend()
    report.add_figure(
        "figures/stationary-distribution.png",
        "Stationary distribution of active firms",
        fig3,
        description=(
            "The invariant distribution is tightly centered because free entry offsets exits "
            "near the profitable range. The black markers show a long simulation from the same "
            f"Markov chain. The maximum simulation gap is **{max_dist_gap:.2e}** after burn-in."
        ),
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
        "Simulated market structure and turnover flows",
        fig4,
        description=(
            "The simulated path shows the same object over time. The firm count stays near the "
            "invariant mean. The lower panel shows the turnover events that prevent absorption. "
            "With this calibration, turnover is modest."
        ),
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
            "Expected incumbent exit probability",
            "Expected exits (firms/period)",
            "Expected entry (firms/period)",
            "Max stationary simulation gap",
            "VFI iterations",
        ],
        "Value": [
            f"{expected_N:.2f}",
            f"{std_N:.2f}",
            f"{mode_N}",
            f"{N_zero_profit:.1f}",
            f"{profits_at_mean:.3f}",
            f"{profits_at_mean - f:.3f}",
            f"{expected_exit_rate:.4f}",
            f"{expected_exits:.3f}",
            f"{expected_entry:.2f}",
            f"{max_dist_gap:.2e}",
            f"{info['iterations']}",
        ],
    }
    df_stats = pd.DataFrame(stats_data)
    report.add_table("tables/equilibrium-statistics.csv", "Equilibrium Statistics", df_stats,
        description=(
            "Expected market size lies below the static zero-profit count. Entrants must "
            "recover sunk cost $K$ plus the operating cost. Incumbents still have continuation "
            "value, so exit remains smooth."
        ))

    # --- Table 2: Value and Policies by N ---
    sample_N = np.array([1, 2, 3, 5, 7, 10, 15, 20, 25, 30])
    sample_N = sample_N[sample_N <= N_max]
    detail_data = {
        "N": [str(n) for n in sample_N],
        "Profit pi(N)": [f"{cournot_profit(n, a, b, c):.3f}" for n in sample_N],
        "Net profit pi-f": [f"{cournot_profit(n, a, b, c) - f:.3f}" for n in sample_N],
        "V(N)": [f"{V[n - 1]:.3f}" for n in sample_N],
        "Exit prob": [f"{exit_prob[n - 1]:.4f}" for n in sample_N],
        "Expected entry": [f"{entry_count[n - 1]:.2f}" for n in sample_N],
    }
    df_detail = pd.DataFrame(detail_data)
    report.add_table(
        "tables/value-by-N.csv",
        "Value Function and Policies at Selected Market Structures",
        df_detail,
        description=(
            "Thin markets have high incumbent value and attract entrants. Crowded markets have "
            "lower profits and higher exit risk. Entry shuts down before incumbents are certain "
            "to leave."
        ),
    )

    report.add_takeaway(
        "The entry and exit conditions separate. Static profits show whether a firm covers the "
        "operating cost. Dynamic values show whether keeping the incumbency option is "
        "worthwhile. A sunk entry cost creates a band where incumbents stay and entrants wait. "
        "That band makes firm counts persistent in Ericson-Pakes style IO models."
    )

    report.add_references([
        "Ericson, R. and Pakes, A. (1995). Markov-perfect industry dynamics: A framework for empirical work. *Review of Economic Studies*, 62(1):53-82.",
        "Hopenhayn, H. (1992). Entry, exit, and firm dynamics in long run equilibrium. *Econometrica*, 60(5):1127-1150.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
