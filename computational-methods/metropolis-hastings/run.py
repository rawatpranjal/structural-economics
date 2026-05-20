#!/usr/bin/env python3
"""Posterior sampling: Beta-Binomial conjugate updates and random-walk Metropolis-Hastings.

The tutorial covers two methods. The first is conjugate Bayes on a
Beta-Binomial model, where the posterior is a Beta distribution in closed
form. It is the canonical first Bayesian example and the controlled
sanity check for the Markov-chain sampler that follows. The second is
random-walk Metropolis-Hastings on a two-component Gaussian mixture, a
stand-in for a structural posterior with two parameter regimes. Closed-form
moments do not exist for the mixture; the sampler is the only way to compute
posterior averages, and the diagnostics earn their keep.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import betainc, gammaln, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


# =============================================================================
# Method 1: Beta-Binomial conjugate model
# =============================================================================
ALPHA_PRIOR = 2.0
BETA_PRIOR = 2.0
N_TRIALS = 20
K_SUCCESSES = 14
ALPHA_POST = ALPHA_PRIOR + K_SUCCESSES
BETA_POST = BETA_PRIOR + N_TRIALS - K_SUCCESSES
POST_MEAN = ALPHA_POST / (ALPHA_POST + BETA_POST)
POST_VAR = (ALPHA_POST * BETA_POST) / (
    (ALPHA_POST + BETA_POST) ** 2 * (ALPHA_POST + BETA_POST + 1)
)


def beta_log_pdf(theta: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Log density of Beta(alpha, beta) for theta in (0, 1)."""
    theta = np.asarray(theta, dtype=float)
    log_norm = gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta)
    return log_norm + (alpha - 1.0) * np.log(theta) + (beta - 1.0) * np.log(1.0 - theta)


def beta_pdf(theta: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return np.exp(beta_log_pdf(theta, alpha, beta))


def beta_binomial_log_posterior(theta: float) -> float:
    """Unnormalized log posterior of theta under Beta(ALPHA_PRIOR, BETA_PRIOR) prior
    and Binomial(N_TRIALS, theta) likelihood with K_SUCCESSES observed."""
    if theta <= 0.0 or theta >= 1.0:
        return -np.inf
    return (ALPHA_POST - 1.0) * np.log(theta) + (BETA_POST - 1.0) * np.log(1.0 - theta)


def random_walk_mh_1d(
    log_target_fn,
    n_draws: int,
    proposal_step: float,
    seed: int,
    start: float,
    bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, float]:
    """One-dimensional Gaussian random-walk Metropolis-Hastings.

    Proposals outside `bounds` are rejected without evaluating the log target,
    which keeps the target on its support and the acceptance ratio well defined.
    """
    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws, dtype=float)
    current = float(start)
    current_logp = float(log_target_fn(current))
    accepted = 0
    draws[0] = current
    for t in range(1, n_draws):
        proposal = current + proposal_step * rng.normal()
        if bounds is not None and (proposal <= bounds[0] or proposal >= bounds[1]):
            draws[t] = current
            continue
        proposal_logp = float(log_target_fn(proposal))
        log_alpha = proposal_logp - current_logp
        if np.log(rng.uniform()) <= min(0.0, log_alpha):
            current = proposal
            current_logp = proposal_logp
            accepted += 1
        draws[t] = current
    return draws, accepted / (n_draws - 1)


# =============================================================================
# Method 2: Two-regime Gaussian mixture posterior (existing material)
# =============================================================================
MU1 = np.array([1.5, 1.5])
MU2 = np.array([-1.5, -1.5])
SIGMA = np.array([[1.0, 0.5], [0.5, 1.0]])
MIXING_PROB = 0.5
INV_SIGMA = np.linalg.inv(SIGMA)
LOG_DET_SIGMA = float(np.linalg.slogdet(SIGMA)[1])
LOG_2PI = float(np.log(2.0 * np.pi))
TRUE_MEAN = MIXING_PROB * MU1 + (1.0 - MIXING_PROB) * MU2
TRUE_COV = SIGMA + MIXING_PROB * (1.0 - MIXING_PROB) * np.outer(MU1 - MU2, MU1 - MU2)


def log_mvn(points: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Evaluate a bivariate normal log density at one or many points."""
    x = np.asarray(points, dtype=float)
    diff = x - mean
    quad = np.einsum("...i,ij,...j->...", diff, INV_SIGMA, diff)
    return -0.5 * (2.0 * LOG_2PI + LOG_DET_SIGMA + quad)


def log_target(points: np.ndarray) -> np.ndarray:
    """Evaluate the log density of the two-component Gaussian mixture."""
    log_components = np.stack(
        [
            np.log(MIXING_PROB) + log_mvn(points, MU1),
            np.log(1.0 - MIXING_PROB) + log_mvn(points, MU2),
        ],
        axis=0,
    )
    return logsumexp(log_components, axis=0)


def random_walk_mh(
    n_draws: int,
    proposal_step: float,
    seed: int,
    start: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Run a Gaussian random-walk Metropolis-Hastings chain in two dimensions."""
    rng = np.random.default_rng(seed)
    draws = np.empty((n_draws, 2), dtype=float)
    current = np.array([10.0, -10.0]) if start is None else np.asarray(start, dtype=float).copy()
    current_logp = float(log_target(current))
    accepted = 0
    draws[0] = current

    for t in range(1, n_draws):
        proposal = current + proposal_step * rng.normal(size=2)
        proposal_logp = float(log_target(proposal))
        log_acceptance = proposal_logp - current_logp
        if np.log(rng.uniform()) <= min(0.0, log_acceptance):
            current = proposal
            current_logp = proposal_logp
            accepted += 1
        draws[t] = current

    return draws, accepted / (n_draws - 1)


def mode_labels(draws: np.ndarray) -> np.ndarray:
    """Assign each draw to its nearest target component."""
    d1 = np.sum((draws - MU1) ** 2, axis=1)
    d2 = np.sum((draws - MU2) ** 2, axis=1)
    return np.where(d1 <= d2, 1, -1)


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute sample autocorrelation up to max_lag."""
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return np.ones(max_lag + 1)
    acf = np.empty(max_lag + 1, dtype=float)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = float(np.dot(x[:-lag], x[lag:]) / denom)
    return acf


def effective_sample_size(series: np.ndarray, max_lag: int = 300) -> float:
    """Estimate effective sample size from positive autocorrelations."""
    acf = autocorrelation(series, min(max_lag, len(series) - 2))
    positive = []
    for value in acf[1:]:
        if value <= 0.0:
            break
        positive.append(value)
    tau = 1.0 + 2.0 * float(np.sum(positive))
    return float(len(series) / max(tau, 1.0))


def chain_summary(chains: dict[float, tuple[np.ndarray, float]], burn: int) -> pd.DataFrame:
    """Create a tuning summary for proposal scales."""
    rows = []
    for proposal_step, (draws, acceptance) in chains.items():
        kept = draws[burn:]
        labels = mode_labels(kept)
        switches = int(np.sum(labels[1:] != labels[:-1]))
        mean_error = float(np.linalg.norm(kept.mean(axis=0) - TRUE_MEAN))
        rows.append(
            {
                "Proposal step": proposal_step,
                "Acceptance rate": f"{acceptance:.3f}",
                "Mode switches": switches,
                "Mean error": f"{mean_error:.3f}",
                "ESS theta1": f"{effective_sample_size(kept[:, 0]):.0f}",
                "ESS theta2": f"{effective_sample_size(kept[:, 1]):.0f}",
            }
        )
    return pd.DataFrame(rows)


def make_density_grid(limit: float = 5.0, n: int = 180) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, Y, Z arrays for target-density contours."""
    grid = np.linspace(-limit, limit, n)
    x_grid, y_grid = np.meshgrid(grid, grid)
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    density = np.exp(log_target(points)).reshape(x_grid.shape)
    return x_grid, y_grid, density


def main() -> None:
    setup_style()

    # -------------------------------------------------------------------------
    # Method 1: Beta-Binomial conjugate model
    # -------------------------------------------------------------------------
    n_conj_draws = 20_000
    burn_conj = 1_000
    proposal_conj = 0.10
    bb_draws, bb_acceptance = random_walk_mh_1d(
        beta_binomial_log_posterior,
        n_draws=n_conj_draws,
        proposal_step=proposal_conj,
        seed=20260510,
        start=0.5,
        bounds=(0.0, 1.0),
    )
    bb_kept = bb_draws[burn_conj:]
    bb_mean_empirical = float(np.mean(bb_kept))
    bb_var_empirical = float(np.var(bb_kept))
    bb_mean_error = abs(bb_mean_empirical - POST_MEAN)
    bb_var_error = abs(bb_var_empirical - POST_VAR)

    # -------------------------------------------------------------------------
    # Method 2: Two-regime Gaussian mixture
    # -------------------------------------------------------------------------
    n_draws = 12_000
    burn = 1_000
    proposal_steps = [0.15, 0.6, 2.0]
    chains = {
        step: random_walk_mh(n_draws=n_draws, proposal_step=step, seed=609 + i)
        for i, step in enumerate(proposal_steps)
    }
    summary = chain_summary(chains, burn=burn)
    main_step = 0.6
    main_draws, main_acceptance = chains[main_step]
    kept = main_draws[burn:]

    print("Method 1: Beta-Binomial conjugate")
    print(f"  closed-form posterior: Beta({ALPHA_POST:.0f}, {BETA_POST:.0f})")
    print(f"  analytical mean = {POST_MEAN:.4f}, variance = {POST_VAR:.5f}")
    print(f"  MH empirical mean = {bb_mean_empirical:.4f} (error {bb_mean_error:.4f})")
    print(f"  MH empirical variance = {bb_var_empirical:.5f} (error {bb_var_error:.5f})")
    print(f"  MH acceptance rate = {bb_acceptance:.3f}")
    print()
    print("Method 2: Two-regime mixture proposal comparison")
    for _, row in summary.iterrows():
        print(
            f"  step={row['Proposal step']}: accept={row['Acceptance rate']} "
            f"switches={row['Mode switches']} mean_error={row['Mean error']}"
        )

    # =========================================================================
    # Results: Method 1 conjugate model
    # =========================================================================
    theta_grid = np.linspace(1e-3, 1 - 1e-3, 600)
    prior_pdf = beta_pdf(theta_grid, ALPHA_PRIOR, BETA_PRIOR)
    posterior_pdf = beta_pdf(theta_grid, ALPHA_POST, BETA_POST)

    fig0, ax0 = plt.subplots(figsize=(8, 5))
    ax0.plot(theta_grid, prior_pdf, color="tab:gray", linestyle="--", linewidth=1.5,
             label=fr"Prior $\mathrm{{Beta}}({ALPHA_PRIOR:.0f}, {BETA_PRIOR:.0f})$")
    ax0.plot(theta_grid, posterior_pdf, color="tab:blue", linewidth=2,
             label=fr"Analytical posterior $\mathrm{{Beta}}({ALPHA_POST:.0f}, {BETA_POST:.0f})$")
    ax0.hist(bb_kept, bins=50, density=True, alpha=0.45, color="tab:orange",
             label=fr"MH histogram ({(n_conj_draws - burn_conj):,} draws)")
    ax0.axvline(POST_MEAN, color="tab:red", linestyle=":", linewidth=1.2,
                label=fr"Analytical mean $= {POST_MEAN:.4f}$")
    ax0.axvline(bb_mean_empirical, color="tab:purple", linestyle=":", linewidth=1.2,
                label=fr"MH mean $= {bb_mean_empirical:.4f}$")
    ax0.set_xlabel(r"$\theta$")
    ax0.set_ylabel("Density")
    ax0.set_title("Beta-Binomial conjugate model: analytical posterior vs Metropolis-Hastings histogram")
    ax0.legend(loc="upper left", fontsize=9)
    save_figure(fig0, "figures/conjugate-posterior.png", dpi=150)

    # Analytical tail probability via the regularized incomplete Beta function.
    p_gt_half_analytical = float(1.0 - betainc(ALPHA_POST, BETA_POST, 0.5))
    p_gt_half_mh = float(np.mean(bb_kept > 0.5))
    bb_summary = pd.DataFrame([
        {
            "Quantity": "Posterior mean",
            "Analytical": f"{POST_MEAN:.4f}",
            "MH empirical": f"{bb_mean_empirical:.4f}",
            "Absolute error": f"{bb_mean_error:.4f}",
        },
        {
            "Quantity": "Posterior variance",
            "Analytical": f"{POST_VAR:.5f}",
            "MH empirical": f"{bb_var_empirical:.5f}",
            "Absolute error": f"{bb_var_error:.5f}",
        },
        {
            "Quantity": "Posterior P(theta > 0.5)",
            "Analytical": f"{p_gt_half_analytical:.4f}",
            "MH empirical": f"{p_gt_half_mh:.4f}",
            "Absolute error": f"{abs(p_gt_half_analytical - p_gt_half_mh):.4f}",
        },
    ])
    Path("tables").mkdir(parents=True, exist_ok=True)
    bb_summary.to_csv("tables/conjugate-summary.csv", index=False)

    # =========================================================================
    # Results: Method 2 mixture posterior
    # =========================================================================
    x_grid, y_grid, density = make_density_grid()
    fig1, ax1 = plt.subplots(figsize=(7.2, 6.2))
    ax1.contour(x_grid, y_grid, density, levels=18, cmap="viridis", alpha=0.9)
    path = main_draws[burn : burn + 4_000 : 4]
    ax1.plot(path[:, 0], path[:, 1], color="tab:blue", alpha=0.45, linewidth=1.0)
    ax1.scatter(kept[::20, 0], kept[::20, 1], s=8, color="tab:orange", alpha=0.35, label="Kept draws")
    ax1.scatter(
        [MU1[0], MU2[0]],
        [MU1[1], MU2[1]],
        marker="*",
        s=180,
        color="black",
        label="Component means",
    )
    ax1.set_xlim(-5.0, 5.0)
    ax1.set_ylim(-5.0, 5.0)
    ax1.set_xlabel(r"$\theta_1$")
    ax1.set_ylabel(r"$\theta_2$")
    ax1.set_title(f"Random-walk path on the mixture target, proposal step {main_step}")
    ax1.legend(loc="upper left")
    save_figure(fig1, "figures/mh-walk.png", dpi=150)

    fig2, axes2 = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    for dim, ax in enumerate(axes2):
        ax.plot(main_draws[:, dim], color="tab:blue", linewidth=0.8)
        ax.axhline(TRUE_MEAN[dim], color="black", linestyle=":", linewidth=1.2, label="Target mean")
        ax.axvline(burn, color="crimson", linestyle="--", linewidth=1.0, label="Burn-in")
        ax.set_ylabel(rf"$\theta_{dim + 1}$")
        ax.legend(loc="upper right")
    axes2[-1].set_xlabel("Draw")
    axes2[0].set_title("Trace plots for the mixture target")
    save_figure(fig2, "figures/trace-plots.png", dpi=150)

    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4.6))
    for proposal_step, (draws, _) in chains.items():
        retained = draws[burn:]
        cumulative_mean = np.cumsum(retained[:, 0]) / np.arange(1, len(retained) + 1)
        acf = autocorrelation(retained[:, 0], 120)
        axes3[0].plot(cumulative_mean, label=f"step {proposal_step}")
        axes3[1].plot(np.arange(len(acf)), acf, label=f"step {proposal_step}")
    axes3[0].axhline(TRUE_MEAN[0], color="black", linestyle=":", linewidth=1.0)
    axes3[0].set_xlabel("Retained draw")
    axes3[0].set_ylabel(r"Cumulative mean of $\theta_1$")
    axes3[0].set_title("Running mean")
    axes3[1].axhline(0.0, color="black", linewidth=0.8)
    axes3[1].set_xlabel("Lag")
    axes3[1].set_ylabel("Autocorrelation")
    axes3[1].set_title("Autocorrelation")
    axes3[1].legend()
    fig3.tight_layout()
    save_figure(fig3, "figures/tuning-diagnostics.png", dpi=150)

    summary.to_csv("tables/proposal-comparison.csv", index=False)

    print(f"\nGenerated: figures + tables")

    save_thumbnail("figures/conjugate-posterior.png", "figures/thumb.png")


if __name__ == "__main__":
    main()
