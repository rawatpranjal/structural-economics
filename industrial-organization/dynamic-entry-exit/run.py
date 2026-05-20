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
from lib.plotting import setup_style, save_figure, save_thumbnail


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

    # Exit probability carried as a state variable alongside V. Each VFI sweep
    # uses the PREVIOUS sweep's exit_prob for the rivals' stay probability; V
    # and exit_prob then converge jointly to the same fixed point, so the
    # converged p_exit(N) and V(N) share the identical Delta(N).
    exit_prob = np.zeros(n_states)
    entry_count = np.zeros(n_states)

    # Dampening factor for stability
    dampen = 0.3

    for iteration in range(1, max_iter + 1):
        V_new = np.zeros(n_states)
        exit_prob_new = np.zeros(n_states)
        entry_count_new = np.zeros(n_states)

        for i in range(n_states):
            N = N_grid[i]
            pi_N = profits[i]

            # Delta(N) = pi(N) - f + beta * E[V(N') | N, stay], where E[V(N')]
            # integrates over the Binomial(N-1, 1 - p_exit) survivor count of
            # the OTHER N-1 rivals. The rivals' stay probability is taken from
            # the previous sweep's exit_prob, breaking the within-sweep
            # circularity. _exit_prob returns the SAME Delta(N) it used (via
            # EV), so V_new and exit_prob are computed from the identical
            # Delta(N).
            p_exit_i, n_enter_current, EV = _exit_prob(
                N, profits, f, beta, V, sigma_eps, K, N_max, exit_prob[i]
            )

            # Value of staying (deterministic component): the SAME Delta(N)
            # that _exit_prob used to compute p_exit above.
            u_stay = pi_N - f + beta * EV

            # Inclusive value (expected value integrating over logistic shock):
            # V(N) = sigma * log(exp(u_stay/sigma) + exp(0/sigma))
            #       = sigma * log(1 + exp(u_stay/sigma))
            # This is the "log-sum" formula from McFadden (1978)
            V_new[i] = sigma_eps * np.logaddexp(u_stay / sigma_eps, 0.0)
            exit_prob_new[i] = p_exit_i
            entry_count_new[i] = n_enter_current

        # Dampened update. The dampened step is 0.3 * (V_new - V); the honest
        # convergence diagnostic is the UNDAMPENED sup-norm residual, since the
        # fixed point is V = V_new, not V = V_update.
        V_update = dampen * V_new + (1.0 - dampen) * V
        error = np.max(np.abs(V_new - V))

        if iteration % 100 == 0:
            print(f"  VFI iteration {iteration:4d}, error = {error:.2e}")

        V = V_update
        exit_prob = exit_prob_new
        entry_count = entry_count_new

        if error < tol:
            print(f"  VFI converged in {iteration} iterations (error = {error:.2e})")
            break

    info = {"iterations": iteration, "converged": error < tol, "error": error}
    return V, exit_prob, entry_count, N_grid, info


def _continuation_value(N, p_stay_others, n_enter, V, N_max):
    """Expected next-period value for an incumbent conditional on staying.

    Integrates V over the Binomial(N-1, p_stay_others) survivor count of the
    OTHER N-1 rivals. The focal firm always counts as one survivor, and
    n_enter entrants are added at the current-state entry rule:

        E[V(N') | N, stay]
            = sum_{s=0}^{N-1} Binomial(s; N-1, p_stay_others)
                              * V(min{s + 1 + n_enter, N_max}).
    """
    EV = 0.0
    for s in range(N):  # s = survivors among the other N-1 rivals
        prob_s = binom.pmf(s, N - 1, p_stay_others) if N > 1 else (1.0 if s == 0 else 0.0)
        if prob_s < 1e-15:
            continue
        N_surv = s + 1  # focal firm stays => +1
        N_next = min(N_surv + n_enter, N_max)
        EV += prob_s * V[N_next - 1]
    return EV


def _exit_prob(N, profits, f, beta, V, sigma_eps, K, N_max, p_exit_rivals):
    """Compute the equilibrium exit probability at state N.

    The logistic exit rule is p_exit(N) = 1 / (1 + exp(Delta(N)/sigma_eps)),
    where Delta(N) = pi(N) - f + beta * E[V(N') | N, stay] is the SAME
    deterministic stay surplus that drives the log-sum value update. The
    continuation value E[V(N')] integrates over the Binomial(N-1, 1 - p_exit)
    survivor distribution of the OTHER N-1 rivals.

    By symmetry every rival uses the same exit rule, so p_exit(N) is the fixed
    point of the map p -> 1/(1+exp(Delta(N; p)/sigma)). That fixed point is
    solved jointly with the value function: the caller passes the previous VFI
    sweep's exit probability ``p_exit_rivals`` for the rivals, and p_exit(N)
    converges to p_exit_rivals as the VFI loop converges. This avoids a nested
    scalar fixed point inside every VFI sweep.

    Returns the triple (p_exit, n_enter, EV), where n_enter is the free-entry
    count and EV is the continuation value E[V(N') | N, stay]. EV is returned
    so the caller's log-sum value update uses the identical Delta(N).
    """
    pi_N = profits[N - 1]

    p_stay_others = 1.0 - p_exit_rivals
    expected_surv = max(0, int(np.round(N * p_stay_others)))
    n_enter = _free_entry_count(
        expected_surv, profits, f, beta, V, K, N_max, sigma_eps
    )
    EV = _continuation_value(N, p_stay_others, n_enter, V, N_max)
    delta = pi_N - f + beta * EV
    p_exit = 1.0 / (1.0 + np.exp(delta / sigma_eps))
    return p_exit, n_enter, EV


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

    setup_style()

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
    save_figure(fig1, "figures/value-function.png", dpi=150)

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
    save_figure(fig2, "figures/entry-exit-probabilities.png", dpi=150)

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
    save_figure(fig3, "figures/stationary-distribution.png", dpi=150)

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
    save_figure(fig4, "figures/simulated-market.png", dpi=150)

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
    Path("tables/equilibrium-statistics.csv").parent.mkdir(parents=True, exist_ok=True)
    df_stats.to_csv("tables/equilibrium-statistics.csv", index=False)

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
    Path("tables/value-by-N.csv").parent.mkdir(parents=True, exist_ok=True)
    df_detail.to_csv("tables/value-by-N.csv", index=False)

    save_thumbnail("figures/value-function.png", "figures/thumb.png")
    print("\nFigures and tables written.")


if __name__ == "__main__":
    main()
