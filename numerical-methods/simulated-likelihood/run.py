#!/usr/bin/env python3
"""Simulated Maximum Likelihood, Common Random Numbers, and Halton Sequences.

Estimates sigma in a one-parameter mixed binary logit by SML under three draw
schemes (pseudo-random, Halton base-2, scrambled Sobol) and three draw counts
(R = 50, 200, 1000). Monte Carlo bias and variance are compared across schemes.

Panel setup: N individuals each face T repeated binary choices with a fixed
beta_i ~ N(mu, sigma^2). Repeated choices identify sigma via within-individual
variation across choice occasions.

Seed scheme:
  - x covariates:  np.random.default_rng(X_SEED)
  - MC trial data: np.random.default_rng(BASE_SEED + trial) for each trial
  - Pseudo draws:  np.random.default_rng(DRAW_SEED + R) -- deterministic per R
  - Halton:        deterministic (sequence depends only on R)
  - Sobol:         Sobol(scramble=True, seed=DRAW_SEED + R) per R
"""
import sys
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import norm, qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N: int = 100              # individuals
T_OBS: int = 5            # repeated binary choices per individual
TRUE_MU: float = 1.0
TRUE_SIGMA: float = 0.8
R_GRID: list[int] = [50, 200, 1000]
MC_TRIALS: int = 200      # 200 trials per spec; ~120s total
X_SEED: int = 42
BASE_SEED: int = 1000     # MC trial i uses BASE_SEED + i
DRAW_SEED: int = 9999     # draw scheme seeds indexed by DRAW_SEED + R
SCHEMES: tuple[str, ...] = ("pseudo", "halton", "sobol")
CLIP_LOW: float = 1e-14   # floor for clipped log-probabilities


# ---------------------------------------------------------------------------
# Draw schemes (standard normal)
# ---------------------------------------------------------------------------

def draws_pseudo(R: int, seed: int) -> np.ndarray:
    """Standard normal pseudo-random draws of length R."""
    return np.random.default_rng(seed).standard_normal(R)


def _halton_base2(k: int) -> float:
    """Halton radical inverse in base 2 for integer k >= 1."""
    result = 0.0
    denominator = 1.0
    n = k
    while n > 0:
        denominator *= 2
        result += (n % 2) / denominator
        n //= 2
    return result


def draws_halton(R: int) -> np.ndarray:
    """1D Halton sequence in base 2, mapped to standard normal via norm.ppf.

    k runs from 1 to R; each point is the bit-reversed binary representation
    of k scaled to (0, 1). Clipping keeps norm.ppf away from the boundary.
    """
    u = np.array([_halton_base2(k) for k in range(1, R + 1)])
    u = np.clip(u, 1e-10, 1 - 1e-10)
    return norm.ppf(u)


def draws_sobol(R: int, seed: int) -> np.ndarray:
    """Scrambled Sobol sequence (1D) mapped to standard normal.

    Sobol has best low-discrepancy properties at power-of-2 sample counts,
    so we draw the next power of 2 >= R and trim to R.
    """
    sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
    n_pow2 = int(2 ** np.ceil(np.log2(max(R, 2))))
    u = sampler.random(n_pow2)[:R, 0]
    u = np.clip(u, 1e-10, 1 - 1e-10)
    return norm.ppf(u)


# ---------------------------------------------------------------------------
# Data simulation
# ---------------------------------------------------------------------------

def simulate_data(
    N: int,
    T: int,
    mu: float,
    sigma: float,
    x: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate binary choices under the mixed binary logit DGP.

    Each individual i has beta_i ~ N(mu, sigma^2) fixed across T occasions.
    Choice probability for occasion t: logistic(beta_i * x_it).
    Returns y of shape (N, T) with entries in {0, 1}.
    """
    beta = rng.normal(mu, sigma, size=N)               # (N,)
    p = 1.0 / (1.0 + np.exp(-beta[:, np.newaxis] * x))  # (N, T)
    return (rng.random((N, T)) < p).astype(np.int8)


# ---------------------------------------------------------------------------
# Simulated log-likelihood with CRN
# ---------------------------------------------------------------------------

def simulated_log_likelihood(
    sigma: float,
    y: np.ndarray,
    x: np.ndarray,
    mu: float,
    z: np.ndarray,
) -> float:
    """Mean simulated log-likelihood at sigma, using fixed draws z (CRN).

    z: standard normal draws of shape (R,).
    beta_r = mu + sigma * z_r  for r = 1..R.

    For individual i, the simulated marginal likelihood integrates over beta_i:
      log P_i(sigma) = log (1/R) sum_r prod_t p_it(beta_r)^{y_it} (1-p_it(beta_r))^{1-y_it}
    Computed stably: first sum log-likelihoods over t for each (r, i), then
    apply logsumexp over r to compute log mean_r, then average over i.
    """
    beta = mu + sigma * z                                # (R,)
    # lin[r, i, t] = beta_r * x_it
    lin = beta[:, np.newaxis, np.newaxis] * x[np.newaxis, :, :]   # (R, N, T)
    log_p1 = -np.logaddexp(0.0, -lin)                   # log logistic(lin), shape (R, N, T)
    log_p0 = -np.logaddexp(0.0,  lin)                   # log(1-logistic(lin)), shape (R, N, T)
    # y has shape (N, T); broadcast along axis 0 (the R axis)
    log_lik_rit = np.where(y[np.newaxis, :, :] == 1, log_p1, log_p0)   # (R, N, T)
    # Sum over T for each (r, i): log P_i | beta_r
    log_lik_ri = log_lik_rit.sum(axis=2)                # (R, N)
    # log mean_r via logsumexp along axis=0, minus log R
    R = len(z)
    log_mean_i = logsumexp(log_lik_ri, axis=0) - np.log(R)   # (N,)
    return float(np.mean(np.clip(log_mean_i, np.log(CLIP_LOW), 0.0)))


def estimate_sigma(
    y: np.ndarray,
    x: np.ndarray,
    mu: float,
    z: np.ndarray,
) -> float:
    """Minimise negative simulated log-likelihood to recover sigma_hat.

    Bounded Brent search over (0.01, 5.0) so sigma stays positive.
    """
    objective = lambda s: -simulated_log_likelihood(s, y, x, mu, z)
    res = minimize_scalar(objective, bounds=(0.01, 5.0), method="bounded")
    return float(res.x)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def run_monte_carlo(
    N: int,
    T: int,
    R_grid: list[int],
    schemes: tuple[str, ...],
    mc_trials: int,
    true_mu: float,
    true_sigma: float,
    x: np.ndarray,
) -> dict[tuple[str, int], np.ndarray]:
    """For each (scheme, R), record sigma_hat across mc_trials.

    Returns dict[(scheme, R)] -> array of length mc_trials.
    Draws are built once per (scheme, R) and reused across MC trials (CRN):
    the same draw set is used for every theta evaluated in every trial,
    making the simulated objective smooth as a function of sigma.
    """
    draw_cache: dict[tuple[str, int], np.ndarray] = {}
    for R in R_grid:
        draw_cache[("pseudo", R)] = draws_pseudo(R, seed=DRAW_SEED + R)
        draw_cache[("halton", R)] = draws_halton(R)
        draw_cache[("sobol",  R)] = draws_sobol(R, seed=DRAW_SEED + R)

    results: dict[tuple[str, int], np.ndarray] = {
        (scheme, R): np.empty(mc_trials)
        for scheme in schemes
        for R in R_grid
    }

    for trial in range(mc_trials):
        rng = np.random.default_rng(BASE_SEED + trial)
        y = simulate_data(N, T, true_mu, true_sigma, x, rng)
        for R in R_grid:
            for scheme in schemes:
                z = draw_cache[(scheme, R)]
                results[(scheme, R)][trial] = estimate_sigma(y, x, true_mu, z)

        if (trial + 1) % 25 == 0:
            print(f"  MC trial {trial + 1}/{mc_trials}")

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

SCHEME_COLORS = {"pseudo": "#4878CF", "halton": "#D65F5F", "sobol": "#6ACC65"}
SCHEME_LABELS = {"pseudo": "Pseudo-random", "halton": "Halton (base 2)", "sobol": "Scrambled Sobol"}


def plot_sampling_distributions(
    results: dict[tuple[str, int], np.ndarray],
    R_ref: int,
    path: str,
) -> None:
    """Boxplots of sigma_hat by scheme at R = R_ref."""
    fig, ax = plt.subplots(figsize=(8, 5))

    data = [results[(scheme, R_ref)] for scheme in SCHEMES]
    labels = [SCHEME_LABELS[s] for s in SCHEMES]
    colors = [SCHEME_COLORS[s] for s in SCHEMES]

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "black", "linewidth": 2},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(
        TRUE_SIGMA, color="black", linestyle="--", linewidth=1.5,
        label=f"True sigma = {TRUE_SIGMA}",
    )
    ax.set_ylabel("Estimated sigma")
    ax.set_title(
        f"Sampling Distribution of sigma_hat  "
        f"(R = {R_ref}, {MC_TRIALS} MC trials, N = {N}, T = {T_OBS})"
    )
    ax.legend(frameon=False)
    save_figure(fig, path, dpi=150)


def plot_bias_variance_vs_R(
    results: dict[tuple[str, int], np.ndarray],
    path: str,
) -> None:
    """Two-panel log-log plot: |bias| and SD of sigma_hat vs R by scheme."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    R_arr = np.array(R_GRID, dtype=float)

    for scheme in SCHEMES:
        bias = np.array([np.mean(results[(scheme, R)]) - TRUE_SIGMA for R in R_GRID])
        sd   = np.array([np.std(results[(scheme, R)], ddof=1)        for R in R_GRID])
        color = SCHEME_COLORS[scheme]
        label = SCHEME_LABELS[scheme]
        axes[0].loglog(R_arr, np.abs(bias), "o-", color=color, label=label)
        axes[1].loglog(R_arr, sd,           "o-", color=color, label=label)

    for ax, ylabel in zip(axes, ["|Bias| of sigma_hat", "SD of sigma_hat"]):
        ax.set_xlabel("Number of draws R")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(frameon=False, fontsize=9)
        ax.set_xticks(R_GRID)
        ax.set_xticklabels([str(r) for r in R_GRID])

    fig.suptitle("Bias-Variance Trade-off by Draw Scheme", fontsize=13)
    fig.tight_layout()
    save_figure(fig, path, dpi=150)


def plot_halton_vs_pseudo_cloud(R: int, path: str) -> None:
    """Side-by-side 2D point clouds: Halton (bases 2 and 3) vs pseudo-random.

    Shows that Halton fills the unit square more uniformly than i.i.d. draws,
    illustrating why quasi-Monte Carlo typically reduces variance.
    """
    def _halton_base3(k: int) -> float:
        result, denominator, n = 0.0, 1.0, k
        while n > 0:
            denominator *= 3
            result += (n % 3) / denominator
            n //= 3
        return result

    u_halton = np.column_stack([
        [_halton_base2(k) for k in range(1, R + 1)],
        [_halton_base3(k) for k in range(1, R + 1)],
    ])

    rng_cloud = np.random.default_rng(DRAW_SEED + 777)
    u_pseudo = rng_cloud.random(size=(R, 2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    titles = [f"Pseudo-random  (R={R})", f"Halton bases 2 & 3  (R={R})"]
    datasets = [u_pseudo, u_halton]
    colors = ["#4878CF", "#D65F5F"]

    for ax, u, title, color in zip(axes, datasets, titles, colors):
        ax.scatter(u[:, 0], u[:, 1], s=8, alpha=0.7, color=color, linewidths=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title(title)
        ax.set_aspect("equal")

    fig.suptitle(
        "Space-Filling: Pseudo-random vs Halton in the Unit Square", fontsize=13
    )
    fig.tight_layout()
    save_figure(fig, path, dpi=150)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_style()
    t0 = time.perf_counter()

    Path("figures").mkdir(parents=True, exist_ok=True)

    rng_x = np.random.default_rng(X_SEED)
    x = rng_x.standard_normal((N, T_OBS))

    print(
        f"Running Monte Carlo: {MC_TRIALS} trials, "
        f"N={N}, T={T_OBS}, R in {R_GRID}"
    )
    results = run_monte_carlo(
        N, T_OBS, R_GRID, SCHEMES, MC_TRIALS, TRUE_MU, TRUE_SIGMA, x
    )

    # Print bias-SD summary table.
    col_w = 13
    header = (
        f"{'Scheme':<18} {'R':>{col_w}} {'Mean':>{col_w}} "
        f"{'Bias':>{col_w}} {'SD':>{col_w}}"
    )
    print()
    print(header)
    print("-" * len(header))
    for scheme in SCHEMES:
        for R in R_GRID:
            arr = results[(scheme, R)]
            mean = float(np.mean(arr))
            bias = mean - TRUE_SIGMA
            sd   = float(np.std(arr, ddof=1))
            print(
                f"{SCHEME_LABELS[scheme]:<18} {R:>{col_w}} "
                f"{mean:>{col_w}.4f} {bias:>{col_w}.4f} {sd:>{col_w}.4f}"
            )
    print()

    plot_sampling_distributions(
        results, R_ref=200, path="figures/sampling-distributions.png"
    )
    plot_bias_variance_vs_R(results, path="figures/bias-variance-vs-R.png")
    plot_halton_vs_pseudo_cloud(R=200, path="figures/halton-vs-pseudo-cloud.png")
    save_thumbnail("figures/bias-variance-vs-R.png", "figures/thumb.png")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s. Figures written to figures/")


if __name__ == "__main__":
    main()
