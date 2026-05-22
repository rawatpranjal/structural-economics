#!/usr/bin/env python3
"""MCMC chain diagnostics: ESS, IAT, and R-hat on a correlated 2D Gaussian.

The target is a correlated bivariate Gaussian with correlation 0.95. The
sampler is random-walk Metropolis-Hastings. Two step-size choices are run:
a tiny step that yields very high autocorrelation, and a near-optimal step
that yields the Roberts-Gelman-Gilks acceptance range.

Three chains per tuning choice are run from dispersed starts. The script
computes lag autocorrelations, the integrated autocorrelation time via
Geyer's monotone-positive truncation, the effective sample size, the
classical Gelman-Rubin R-hat, and the rank-normalised split-R-hat of
Vehtari et al. (2021).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm as norm_dist
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


# =============================================================================
# Target: correlated bivariate Gaussian
# =============================================================================
RHO = 0.95
TARGET_COV = np.array([[1.0, RHO], [RHO, 1.0]])
TARGET_INV = np.linalg.inv(TARGET_COV)
LOG_DET = float(np.linalg.slogdet(TARGET_COV)[1])
LOG_2PI = float(np.log(2.0 * np.pi))


def log_target(point: np.ndarray) -> float:
    """Bivariate normal log density at zero mean."""
    diff = np.asarray(point, dtype=float)
    quad = float(diff @ TARGET_INV @ diff)
    return -0.5 * (2.0 * LOG_2PI + LOG_DET + quad)


# =============================================================================
# Random-walk Metropolis-Hastings
# =============================================================================
def random_walk_mh(
    n_draws: int,
    proposal_step: float,
    seed: int,
    start: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run a two-dimensional Gaussian random-walk chain."""
    rng = np.random.default_rng(seed)
    draws = np.empty((n_draws, 2), dtype=float)
    current = np.asarray(start, dtype=float).copy()
    current_logp = log_target(current)
    accepted = 0
    draws[0] = current
    for t in range(1, n_draws):
        proposal = current + proposal_step * rng.normal(size=2)
        proposal_logp = log_target(proposal)
        log_alpha = proposal_logp - current_logp
        if np.log(rng.uniform()) <= min(0.0, log_alpha):
            current = proposal
            current_logp = proposal_logp
            accepted += 1
        draws[t] = current
    return draws, accepted / (n_draws - 1)


# =============================================================================
# Autocorrelation and Geyer monotone-positive IAT
# =============================================================================
def sample_autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Return sample autocorrelation up to max_lag (lag 0 = 1)."""
    x = np.asarray(series, dtype=float) - np.mean(series)
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return np.ones(max_lag + 1)
    acf = np.empty(max_lag + 1, dtype=float)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = float(np.dot(x[:-lag], x[lag:]) / denom)
    return acf


def geyer_iat(series: np.ndarray, max_lag: int) -> tuple[float, int]:
    """Geyer (1992) initial monotone-positive estimator of the IAT.

    Pair adjacent lags, keep the running minimum, sum until the first
    nonpositive pair. Returns tau and the truncation lag used.
    """
    acf = sample_autocorrelation(series, max_lag)
    pair_sums = []
    for i in range(0, max_lag - 1, 2):
        gamma = acf[i + 1] + acf[i + 2]
        if gamma <= 0:
            break
        pair_sums.append(gamma)
    # Initial monotone sequence: enforce nonincreasing pair sums.
    monotone = []
    running_min = np.inf
    for value in pair_sums:
        running_min = min(running_min, value)
        monotone.append(running_min)
    tau = 1.0 + 2.0 * float(np.sum(monotone))
    cutoff = 2 * len(monotone) + 1 if monotone else 1
    return float(max(tau, 1.0)), int(cutoff)


def effective_sample_size(series: np.ndarray, max_lag: int = 400) -> float:
    tau, _ = geyer_iat(series, min(max_lag, len(series) - 4))
    return float(len(series) / tau)


# =============================================================================
# R-hat: classical and rank-normalised split
# =============================================================================
def classical_rhat(chains: np.ndarray) -> float:
    """Classical Gelman-Rubin R-hat for an (M, T) array of scalar chains."""
    M, T = chains.shape
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)
    W = chain_vars.mean()
    B = T * chain_means.var(ddof=1)
    var_hat = (T - 1) / T * W + B / T
    return float(np.sqrt(var_hat / W))


def split_chains(chains: np.ndarray) -> np.ndarray:
    """Split each chain in half. Returns (2M, T // 2) array."""
    M, T = chains.shape
    half = T // 2
    return np.concatenate([chains[:, :half], chains[:, half : 2 * half]], axis=0)


def rank_normalise(chains: np.ndarray) -> np.ndarray:
    """Pool, rank, then transform to approximate-normal scores."""
    M, T = chains.shape
    flat = chains.reshape(-1)
    ranks = rankdata(flat)
    quantiles = (ranks - 3.0 / 8.0) / (len(flat) + 1.0 / 4.0)
    return norm_dist.ppf(quantiles).reshape(M, T)


def split_rank_rhat(chains: np.ndarray) -> float:
    """Rank-normalised split-R-hat (Vehtari et al. 2021)."""
    return classical_rhat(rank_normalise(split_chains(chains)))


# =============================================================================
# Experiment
# =============================================================================
N_DRAWS = 8_000
BURN = 1_000
STARTS = np.array([[3.0, -3.0], [-3.0, 3.0], [0.0, 0.0]])
STEPS = {"tiny": 0.05, "optimal": 0.9}  # 2D RWM rule-of-thumb optimal ~ 0.9


def run_all_chains() -> dict:
    """Return chains[(step_label, chain_index)] = (draws, acceptance)."""
    chains = {}
    for step_label, step in STEPS.items():
        for j, start in enumerate(STARTS):
            seed = 20260522 + 1000 * (step_label == "optimal") + j
            chains[(step_label, j)] = random_walk_mh(
                n_draws=N_DRAWS, proposal_step=step, seed=seed, start=start
            )
    return chains


def collect_chains_array(
    chains: dict, step_label: str, dim: int, burn: int
) -> np.ndarray:
    """Return (M, T-burn) array of one coordinate post-burn-in."""
    rows = []
    for j in range(len(STARTS)):
        draws, _ = chains[(step_label, j)]
        rows.append(draws[burn:, dim])
    return np.array(rows)


# =============================================================================
# Plots
# =============================================================================
def plot_trace(chains: dict) -> None:
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6.5), sharex=True)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for col, step_label in enumerate(["tiny", "optimal"]):
        for dim in range(2):
            ax = axes[dim, col]
            for j in range(len(STARTS)):
                draws, accept = chains[(step_label, j)]
                ax.plot(
                    draws[:, dim],
                    color=colors[j],
                    linewidth=0.7,
                    alpha=0.85,
                    label=f"Chain {j+1} (accept {accept:.2f})" if dim == 0 else None,
                )
            ax.axvline(BURN, color="crimson", linestyle="--", linewidth=0.9)
            ax.set_ylabel(rf"$\theta_{dim + 1}$")
            ax.set_ylim(-5, 5)
            if dim == 0:
                ax.set_title(f"{step_label.capitalize()} step (s = {STEPS[step_label]})")
                ax.legend(loc="upper right", fontsize=8)
        axes[1, col].set_xlabel("Draw")
    fig.suptitle("Trace plots: tiny step (pathological) vs near-optimal step")
    fig.tight_layout()
    save_figure(fig, "figures/trace-plots.png", dpi=150)


def plot_autocorrelation(chains: dict) -> None:
    max_lag = 200
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.6), sharey=True)
    for ax, step_label in zip(axes, ["tiny", "optimal"]):
        for j in range(len(STARTS)):
            draws, _ = chains[(step_label, j)]
            series = draws[BURN:, 0]
            acf = sample_autocorrelation(series, max_lag)
            _, cutoff = geyer_iat(series, max_lag)
            ax.plot(np.arange(max_lag + 1), acf, linewidth=1.2,
                    label=f"Chain {j+1}, Geyer cutoff = {cutoff}")
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_xlabel("Lag")
        ax.set_title(f"{step_label.capitalize()} step (s = {STEPS[step_label]})")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel(r"Autocorrelation of $\theta_1$")
    fig.suptitle("Autocorrelation decay with Geyer monotone-positive truncation")
    fig.tight_layout()
    save_figure(fig, "figures/autocorrelation-decay.png", dpi=150)


def plot_rhat_trajectory(chains: dict) -> None:
    grid = np.unique(np.linspace(200, N_DRAWS - BURN, 30).astype(int))
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for step_label, color in zip(["tiny", "optimal"], ["tab:red", "tab:blue"]):
        arr = collect_chains_array(chains, step_label, dim=0, burn=BURN)
        classical = []
        rank_split = []
        for T_use in grid:
            sub = arr[:, :T_use]
            classical.append(classical_rhat(sub))
            rank_split.append(split_rank_rhat(sub))
        ax.plot(grid, classical, color=color, linestyle="--", linewidth=1.6,
                label=f"{step_label} step, classical R-hat")
        ax.plot(grid, rank_split, color=color, linestyle="-", linewidth=1.8,
                label=f"{step_label} step, rank-normalised split R-hat")
    ax.axhline(1.01, color="black", linewidth=0.8, linestyle=":",
               label="Vehtari threshold 1.01")
    ax.set_xlabel("Post-burn-in chain length T")
    ax.set_ylabel(r"$\hat R$")
    ax.set_title(r"$\hat R$ trajectory vs chain length")
    ax.legend(loc="upper right", fontsize=8)
    save_figure(fig, "figures/r-hat-trajectory.png", dpi=150)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    setup_style()
    chains = run_all_chains()

    rows = []
    for step_label in ["tiny", "optimal"]:
        arr_theta1 = collect_chains_array(chains, step_label, dim=0, burn=BURN)
        arr_theta2 = collect_chains_array(chains, step_label, dim=1, burn=BURN)
        accept_rates = [chains[(step_label, j)][1] for j in range(len(STARTS))]
        ess_per_chain_1 = [effective_sample_size(arr_theta1[j]) for j in range(len(STARTS))]
        ess_per_chain_2 = [effective_sample_size(arr_theta2[j]) for j in range(len(STARTS))]
        tau_1 = [geyer_iat(arr_theta1[j], 400)[0] for j in range(len(STARTS))]
        rows.append(
            {
                "Step": step_label,
                "s": STEPS[step_label],
                "Mean acceptance": float(np.mean(accept_rates)),
                "Mean IAT theta1": float(np.mean(tau_1)),
                "Mean ESS theta1": float(np.mean(ess_per_chain_1)),
                "Mean ESS theta2": float(np.mean(ess_per_chain_2)),
                "Classical R-hat theta1": classical_rhat(arr_theta1),
                "Split rank R-hat theta1": split_rank_rhat(arr_theta1),
                "Classical R-hat theta2": classical_rhat(arr_theta2),
                "Split rank R-hat theta2": split_rank_rhat(arr_theta2),
            }
        )

    summary = pd.DataFrame(rows)
    print("Diagnostics summary")
    for _, row in summary.iterrows():
        print(
            f"  step={row['Step']:<7s} s={row['s']:.2f}: "
            f"accept={row['Mean acceptance']:.3f}, "
            f"IAT={row['Mean IAT theta1']:.1f}, "
            f"ESS(theta1)={row['Mean ESS theta1']:.1f}, "
            f"classical R-hat={row['Classical R-hat theta1']:.4f}, "
            f"split rank R-hat={row['Split rank R-hat theta1']:.4f}"
        )

    Path("tables").mkdir(parents=True, exist_ok=True)
    summary.to_csv("tables/diagnostics-summary.csv", index=False)

    plot_trace(chains)
    plot_autocorrelation(chains)
    plot_rhat_trajectory(chains)

    save_thumbnail("figures/trace-plots.png", "figures/thumb.png")
    print("\nGenerated: figures + tables + thumb")


if __name__ == "__main__":
    main()
