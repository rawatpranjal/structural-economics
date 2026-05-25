#!/usr/bin/env python3
"""Sequential investment under Bayesian learning.

A firm observes noisy signals about a binary project type before deciding
whether to invest. The posterior belief is the state variable for the
finite-horizon stopping problem.

Reference: DeGroot (1970), Roberts and Weitzman (1981), Chamley (2003).
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


def exact_kl_to_truth(
    true_state: str,
    T: int,
    prior_H: float,
    p_red_H: float,
    p_red_L: float,
) -> np.ndarray:
    """KL divergence from true state to posterior, integrated over signal law.

    KL(delta_theta || p_t) = -log p_t(theta*), where theta* is the true state.
    This equals the expected log-loss of the posterior at the truth.
    A lower value means the posterior is closer to the truth.
    Returns array of length T+1 starting at KL(prior).
    """
    p_red = p_red_H if true_state == "H" else p_red_L
    kl = np.zeros(T + 1)
    for t in range(T + 1):
        k_grid = np.arange(t + 1)
        weights = binom.pmf(k_grid, t, p_red)
        posteriors = posterior_from_counts(k_grid, t, prior_H, p_red_H, p_red_L)
        # KL(delta_H || p_t) = -log(p_t(H)) if true state is H
        if true_state == "H":
            log_p = np.log(np.clip(posteriors, 1e-15, 1.0))
        else:
            log_p = np.log(np.clip(1.0 - posteriors, 1e-15, 1.0))
        kl[t] = -float(np.sum(weights * log_p))
    return kl


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
    kl_H = exact_kl_to_truth("H", T, prior_H, p_red_H, p_red_L)
    kl_L = exact_kl_to_truth("L", T, prior_H, p_red_H, p_red_L)

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

    # --- Figure 1 (2x2): belief paths (top) | log-Bayes-factor + KL convergence (bottom) ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 9))
    ax1a, ax1b = axes1[0, 0], axes1[0, 1]
    ax1c, ax1d = axes1[1, 0], axes1[1, 1]

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
    ax1a.set_title("Project is good ($\\theta = H$)")
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
    ax1b.set_title("Project is bad ($\\theta = L$)")
    ax1b.set_ylim(-0.05, 1.05)
    ax1b.legend()

    # Log-Bayes-factor: cumulative evidence for H vs L along exact mean paths
    # log BF_t = sum of per-signal log-likelihood ratios, integrated over signal law
    # For the H-side: E[Lambda_t | theta=H]; for L-side: E[Lambda_t | theta=L]
    log_bf_H = np.zeros(T + 1)
    log_bf_L = np.zeros(T + 1)
    p_red = p_red_H  # true state H
    for t in range(1, T + 1):
        k_grid = np.arange(t + 1)
        lambda_t = (
            k_grid * np.log(p_red_H / p_red_L)
            + (t - k_grid) * np.log((1 - p_red_H) / (1 - p_red_L))
        )
        log_bf_H[t] = float(np.sum(binom.pmf(k_grid, t, p_red_H) * lambda_t))
        log_bf_L[t] = float(np.sum(binom.pmf(k_grid, t, p_red_L) * lambda_t))

    ax1c.plot(periods, log_bf_H, color="steelblue", linewidth=2.0,
              label="$\\theta = H$ (evidence for $H$)")
    ax1c.plot(periods, log_bf_L, color="indianred", linewidth=2.0,
              label="$\\theta = L$ (evidence misleads)")
    ax1c.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    ax1c.set_xlabel("Number of signals")
    ax1c.set_ylabel(r"$\mathbb{E}[\Lambda_t \mid \theta]$")
    ax1c.set_title("Evidence accumulation (log-Bayes-factor)")
    ax1c.legend()

    # KL divergence to truth: posterior concentration
    ax1d.plot(periods, kl_H, color="steelblue", linewidth=2.0,
              label="$\\theta = H$")
    ax1d.plot(periods, kl_L, color="indianred", linewidth=2.0,
              label="$\\theta = L$")
    ax1d.set_xlabel("Number of signals")
    ax1d.set_ylabel(r"$\mathrm{KL}(\delta_{\theta^*} \| p_t)$")
    ax1d.set_title("Posterior concentration (KL to truth)")
    ax1d.legend()

    fig1.tight_layout()
    save_figure(fig1, "figures/belief-evolution.png", dpi=150)

    # --- Figure 2 (1x2): stopping boundary | value-function slices by horizon ---
    # Rerun backward induction to collect V at selected horizons for right panel.
    n_p = 1000
    p_grid_plot = np.linspace(0.001, 0.999, n_p)

    def v_invest_fn(p):
        return p * payoff_invest_H + (1.0 - p) * payoff_invest_L

    V_term = np.maximum(v_invest_fn(p_grid_plot), payoff_wait)
    V_by_horizon = {T_stop: V_term.copy()}
    V_curr = V_term.copy()
    for t in range(T_stop - 1, -1, -1):
        V_new = np.zeros(n_p)
        for i, p in enumerate(p_grid_plot):
            v_stop = max(v_invest_fn(p), payoff_wait)
            p_red_pred = p * p_red_H + (1.0 - p) * p_red_L
            p_after_red = p * p_red_H / p_red_pred
            p_blue = 1.0 - p_red_pred
            p_after_blue = p * (1.0 - p_red_H) / p_blue if p_blue > 0 else p
            v_red = np.interp(p_after_red, p_grid_plot, V_curr)
            v_blue = np.interp(p_after_blue, p_grid_plot, V_curr)
            v_cont = p_red_pred * v_red + p_blue * v_blue
            V_new[i] = max(v_stop, v_cont)
        V_curr = V_new
        if t in {0, 5, 10, 20}:
            V_by_horizon[t] = V_curr.copy()

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
    t_grid = np.arange(T_stop + 1)

    ax2a.fill_between(t_grid, upper_bounds, 1.0, alpha=0.3, color="green", label="Invest")
    ax2a.fill_between(t_grid, 0.0, lower_bounds, alpha=0.3, color="red", label="Don't invest")
    ax2a.fill_between(t_grid, lower_bounds, upper_bounds, alpha=0.2, color="gray",
                      label="Continue observing")
    ax2a.plot(t_grid, upper_bounds, "g-", linewidth=2)
    ax2a.plot(t_grid, lower_bounds, "r-", linewidth=2)

    # Overlay a few belief paths
    for i in range(5):
        path, _ = simulate_belief_path("H", T_stop, prior_H, p_red_H, p_red_L, rng)
        ax2a.plot(np.arange(T_stop + 1), path, "k-", alpha=0.3, linewidth=0.8)

    ax2a.set_xlabel("Period")
    ax2a.set_ylabel("Belief $P(H)$")
    ax2a.set_title("Stopping regions over the belief state")
    ax2a.set_ylim(-0.05, 1.05)
    ax2a.legend(loc="center right")

    # Value-function slices: V_t(p) for t = 0, 5, 10, 20, T (terminal)
    horizon_cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(V_by_horizon)))
    for idx, (t_key, V_slice) in enumerate(sorted(V_by_horizon.items())):
        label = f"$t = {t_key}$" if t_key < T_stop else f"$t = T$ (terminal)"
        ax2b.plot(p_grid_plot, V_slice, color=horizon_cmap[idx], linewidth=1.8, label=label)
    ax2b.plot(p_grid_plot, v_invest_fn(p_grid_plot), color="0.5", linestyle=":",
              linewidth=1.4, label="Action value $A(p)$")
    ax2b.set_xlabel("Belief $P(H)$")
    ax2b.set_ylabel("$V_t(p)$")
    ax2b.set_title("Value function at selected horizons")
    ax2b.set_xlim(0, 1)
    ax2b.legend(loc="upper left", fontsize=8)

    fig2.tight_layout()
    save_figure(fig2, "figures/stopping-boundary.png", dpi=150)

    save_thumbnail("figures/belief-evolution.png", "figures/thumb.png")
    print("\nDone: 2 figures (2x2 + 1x2), thumb reproduced.")


if __name__ == "__main__":
    main()
