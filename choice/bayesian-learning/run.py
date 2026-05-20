#!/usr/bin/env python3
"""Sequential investment under Bayesian learning.

A firm observes noisy signals about a binary project type before deciding
whether to invest. The posterior belief is the state variable for the
finite-horizon stopping problem.

Reference: DeGroot (1970), Chamley (2003).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


# =============================================================================
# Bayesian updating engine
# =============================================================================

def bayesian_update(prior_H: float, signal: int, p_red_H: float, p_red_L: float) -> float:
    """Update belief P(H) after observing a signal (1=red, 0=blue).

    Applies Bayes' rule:
        P(H|s) = P(s|H) * P(H) / [P(s|H)*P(H) + P(s|L)*P(L)]
    """
    lik_H = p_red_H if signal == 1 else (1.0 - p_red_H)
    lik_L = p_red_L if signal == 1 else (1.0 - p_red_L)
    marginal = lik_H * prior_H + lik_L * (1.0 - prior_H)
    posterior = lik_H * prior_H / marginal
    return posterior


def posterior_from_counts(
    red_count: np.ndarray | int,
    T: int,
    prior_H: float,
    p_red_H: float,
    p_red_L: float,
) -> np.ndarray | float:
    """Return P(H | k red signals out of T draws)."""
    if T == 0:
        if np.isscalar(red_count):
            return prior_H
        return np.full_like(np.asarray(red_count, dtype=float), prior_H)

    k = np.asarray(red_count, dtype=float)
    log_prior_odds = np.log(prior_H / (1.0 - prior_H))
    log_lr = (
        k * np.log(p_red_H / p_red_L)
        + (T - k) * np.log((1.0 - p_red_H) / (1.0 - p_red_L))
    )
    posterior = 1.0 / (1.0 + np.exp(-(log_prior_odds + log_lr)))
    if np.isscalar(red_count):
        return float(posterior)
    return posterior


def exact_mean_posterior_path(
    true_state: str,
    T: int,
    prior_H: float,
    p_red_H: float,
    p_red_L: float,
) -> np.ndarray:
    """Integrate the posterior over the binomial signal distribution."""
    p_red = p_red_H if true_state == "H" else p_red_L
    means = np.zeros(T + 1)
    for t in range(T + 1):
        k_grid = np.arange(t + 1)
        posterior = posterior_from_counts(k_grid, t, prior_H, p_red_H, p_red_L)
        means[t] = np.sum(binom.pmf(k_grid, t, p_red) * posterior)
    return means


def simulate_belief_path(true_state: str, T: int, prior_H: float,
                         p_red_H: float, p_red_L: float, rng: np.random.Generator):
    """Simulate a single path of posterior beliefs over T signals.

    Returns:
        beliefs: array of shape (T+1,) starting with the prior
        signals: array of shape (T,) with 1=red, 0=blue
    """
    p_red = p_red_H if true_state == "H" else p_red_L
    signals = rng.binomial(1, p_red, size=T)
    beliefs = np.zeros(T + 1)
    beliefs[0] = prior_H
    for t in range(T):
        beliefs[t + 1] = bayesian_update(beliefs[t], signals[t], p_red_H, p_red_L)
    return beliefs, signals


# =============================================================================
# Optimal stopping
# =============================================================================

def compute_optimal_stopping_boundary(T: int, payoff_invest_H: float,
                                      payoff_invest_L: float,
                                      payoff_wait: float,
                                      p_red_H: float, p_red_L: float):
    """Compute the optimal stopping boundary via backward induction.

    At each period, the agent can:
      - Invest: expected payoff = p*payoff_invest_H + (1-p)*payoff_invest_L
      - Don't invest (stop gathering info): payoff = payoff_wait (=0)
      - Continue: expected value of waiting one more period

    Returns upper and lower belief thresholds for each period.
    """
    # Value of investing at belief p
    def v_invest(p):
        return p * payoff_invest_H + (1.0 - p) * payoff_invest_L

    # Value of not investing
    v_not_invest = payoff_wait

    # At terminal period T, must decide: invest or not
    # Invest if v_invest(p) > v_not_invest => p > threshold
    p_threshold_invest = (payoff_wait - payoff_invest_L) / (payoff_invest_H - payoff_invest_L)

    # Backward induction: at each t, continuation value vs stopping
    # We discretize the belief space
    n_p = 1000
    p_grid = np.linspace(0.001, 0.999, n_p)

    # Terminal value
    V = np.maximum(v_invest(p_grid), v_not_invest)

    upper_bounds = np.zeros(T + 1)
    lower_bounds = np.zeros(T + 1)

    # At terminal period
    upper_bounds[T] = p_threshold_invest
    lower_bounds[T] = p_threshold_invest

    for t in range(T - 1, -1, -1):
        V_new = np.zeros(n_p)
        for i, p in enumerate(p_grid):
            # Value of stopping now
            v_stop = max(v_invest(p), v_not_invest)

            # Value of continuing: expected V(p') after one more signal
            # P(red) = p * p_red_H + (1-p) * p_red_L
            p_red = p * p_red_H + (1.0 - p) * p_red_L
            # Posterior after red signal
            p_after_red = p * p_red_H / p_red
            # Posterior after blue signal
            p_blue = 1.0 - p_red
            p_after_blue = p * (1.0 - p_red_H) / p_blue if p_blue > 0 else p

            v_red = np.interp(p_after_red, p_grid, V)
            v_blue = np.interp(p_after_blue, p_grid, V)
            v_continue = p_red * v_red + p_blue * v_blue

            V_new[i] = max(v_stop, v_continue)

        # Find boundaries where agent switches from continue to stop
        # Upper: invest region
        invest_val = v_invest(p_grid)
        stop_better = (np.maximum(invest_val, v_not_invest) >= V_new - 1e-10)
        # Upper boundary: highest p where continuing is still better
        continue_region = ~stop_better
        if np.any(continue_region):
            indices = np.where(continue_region)[0]
            # Upper bound: where continuation region ends (invest threshold)
            upper_bounds[t] = p_grid[indices[-1]] if len(indices) > 0 else p_threshold_invest
            # Lower bound: where continuation region starts (don't invest threshold)
            lower_bounds[t] = p_grid[indices[0]] if len(indices) > 0 else p_threshold_invest
        else:
            upper_bounds[t] = p_threshold_invest
            lower_bounds[t] = p_threshold_invest

        V = V_new

    return upper_bounds, lower_bounds


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    p_red_H = 0.7         # P(red | state=H)
    p_red_L = 0.3         # P(red | state=L)
    prior_H = 0.5         # Prior belief P(H)
    T = 50                # Number of signals
    n_paths = 200         # Number of simulation paths per state
    seed = 42

    # Optimal stopping payoffs
    payoff_invest_H = 1.0   # Payoff from investing when state is H
    payoff_invest_L = -0.5  # Payoff from investing when state is L
    payoff_wait = 0.0       # Payoff from not investing

    rng = np.random.default_rng(seed)

    # =========================================================================
    # Simulate belief paths
    # =========================================================================
    print("Simulating belief paths...")
    beliefs_H = np.zeros((n_paths, T + 1))  # Paths when true state is H
    beliefs_L = np.zeros((n_paths, T + 1))  # Paths when true state is L

    for i in range(n_paths):
        beliefs_H[i], _ = simulate_belief_path("H", T, prior_H, p_red_H, p_red_L, rng)
        beliefs_L[i], _ = simulate_belief_path("L", T, prior_H, p_red_H, p_red_L, rng)
    exact_mean_H = exact_mean_posterior_path("H", T, prior_H, p_red_H, p_red_L)
    exact_mean_L = exact_mean_posterior_path("L", T, prior_H, p_red_H, p_red_L)

    # =========================================================================
    # Optimal stopping boundary
    # =========================================================================
    print("Computing optimal stopping boundary...")
    T_stop = 30  # Shorter horizon for stopping problem
    upper_bounds, lower_bounds = compute_optimal_stopping_boundary(
        T_stop, payoff_invest_H, payoff_invest_L, payoff_wait, p_red_H, p_red_L
    )

    # =========================================================================
    # Figures
    # =========================================================================
    setup_style()

    # --- Figure 1: Posterior belief evolution ---
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5))

    periods = np.arange(T + 1)
    # Plot subset of paths for clarity
    n_show = 30
    for i in range(n_show):
        ax1a.plot(periods, beliefs_H[i], color="steelblue", alpha=0.2, linewidth=0.8)
    ax1a.plot(periods, np.mean(beliefs_H, axis=0), color="darkblue", linewidth=2.5,
              label="Simulated mean")
    ax1a.plot(periods, exact_mean_H, color="black", linestyle="--", linewidth=1.6,
              label="Exact mean")
    ax1a.axhline(y=1.0, color="black", linestyle="--", alpha=0.3, linewidth=1)
    ax1a.set_xlabel("Number of signals")
    ax1a.set_ylabel("$P(H)$")
    ax1a.set_title("Project is good")
    ax1a.set_ylim(-0.05, 1.05)
    ax1a.legend()

    for i in range(n_show):
        ax1b.plot(periods, beliefs_L[i], color="indianred", alpha=0.2, linewidth=0.8)
    ax1b.plot(periods, np.mean(beliefs_L, axis=0), color="darkred", linewidth=2.5,
              label="Simulated mean")
    ax1b.plot(periods, exact_mean_L, color="black", linestyle="--", linewidth=1.6,
              label="Exact mean")
    ax1b.axhline(y=0.0, color="black", linestyle="--", alpha=0.3, linewidth=1)
    ax1b.set_xlabel("Number of signals")
    ax1b.set_ylabel("$P(H)$")
    ax1b.set_title("Project is bad")
    ax1b.set_ylim(-0.05, 1.05)
    ax1b.legend()

    fig1.suptitle("Posterior beliefs under repeated signals", fontsize=14, fontweight="bold")
    fig1.tight_layout()
    save_figure(fig1, "figures/belief-evolution.png", dpi=150)

    # --- Figure 2: Optimal stopping boundary ---
    fig4, ax4 = plt.subplots()
    t_grid = np.arange(T_stop + 1)

    ax4.fill_between(t_grid, upper_bounds, 1.0, alpha=0.3, color="green", label="Invest")
    ax4.fill_between(t_grid, 0.0, lower_bounds, alpha=0.3, color="red", label="Don't invest")
    ax4.fill_between(t_grid, lower_bounds, upper_bounds, alpha=0.2, color="gray",
                     label="Continue observing")
    ax4.plot(t_grid, upper_bounds, "g-", linewidth=2)
    ax4.plot(t_grid, lower_bounds, "r-", linewidth=2)

    # Overlay a few belief paths
    for i in range(5):
        path, _ = simulate_belief_path("H", T_stop, prior_H, p_red_H, p_red_L, rng)
        ax4.plot(np.arange(T_stop + 1), path, "k-", alpha=0.3, linewidth=0.8)

    ax4.set_xlabel("Period")
    ax4.set_ylabel("Belief $P(H)$")
    ax4.set_title("Stopping regions over the belief state")
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(loc="center right")
    save_figure(fig4, "figures/stopping-boundary.png", dpi=150)

    save_thumbnail("figures/belief-evolution.png", "figures/thumb.png")
    print("\nDone: 2 figures, thumb reproduced.")


if __name__ == "__main__":
    main()
