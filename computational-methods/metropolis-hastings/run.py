#!/usr/bin/env python3
"""Random-walk Metropolis-Hastings for a bimodal target.

This tutorial samples from a mixture of two bivariate normals, which makes
proposal tuning and mode switching visible in trace plots and diagnostics.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


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
    """Run a Gaussian random-walk Metropolis-Hastings chain."""
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

    print("Metropolis-Hastings proposal comparison")
    for _, row in summary.iterrows():
        print(
            f"  step={row['Proposal step']}: accept={row['Acceptance rate']} "
            f"switches={row['Mode switches']} mean_error={row['Mean error']}"
        )

    report = ModelReport(
        "Metropolis-Hastings Sampling Diagnostics",
        "Random-walk MCMC, proposal tuning, and mixing diagnostics on a bimodal target.",
    )

    report.add_overview(
        "Metropolis-Hastings turns a density that is easy to evaluate into draws from that "
        "density. The algorithm is simple: propose a move, compare the target density at the "
        "new and old locations, and sometimes accept a worse move so the chain keeps exploring.\n\n"
        "This tutorial uses the same bimodal mixture as the optimization example. It is "
        "intentionally small enough to plot. Small proposals have high acceptance but move "
        "slowly. Large proposals jump farther but are often rejected. Useful MCMC lives "
        "between those extremes."
    )

    report.add_equations(
        r"""
The target is the same two-component mixture used in the optimization tutorial:

$$
\begin{aligned}
p(\theta)
&= \omega \phi(\theta; \mu_1, \Sigma) \\
&\quad + (1-\omega)\phi(\theta; \mu_2, \Sigma).
\end{aligned}
$$

Given current draw $\theta_t$, a random-walk proposal draws:

$$
\theta^\star = \theta_t + s \eta_t, \qquad \eta_t \sim N(0, I).
$$

Because the proposal is symmetric, the Metropolis-Hastings acceptance probability is:

$$
\alpha(\theta_t,\theta^\star)
= \min\left\{1,\frac{p(\theta^\star)}{p(\theta_t)}\right\}.
$$
"""
    )

    report.add_model_setup(
        f"| Object | Value |\n"
        f"|--------|-------|\n"
        f"| $\\mu_1$ | ({MU1[0]:.1f}, {MU1[1]:.1f}) |\n"
        f"| $\\mu_2$ | ({MU2[0]:.1f}, {MU2[1]:.1f}) |\n"
        f"| $\\Sigma$ | [[1.0, 0.5], [0.5, 1.0]] |\n"
        f"| Mixing probability $\\omega$ | {MIXING_PROB:.1f} |\n"
        f"| Draws | {n_draws:,} |\n"
        f"| Burn-in | {burn:,} |\n"
        f"| Starting point | (10.0, -10.0) |\n"
        f"| Proposal steps | {proposal_steps} |"
    )

    report.add_solution_method(
        "The script evaluates the log target directly and runs a Gaussian random-walk chain. "
        "All acceptance decisions are made in log space to avoid numerical underflow.\n\n"
        "The diagnostics compare three proposal step sizes. For each chain, the code reports "
        "the acceptance rate, number of switches between modes, posterior mean error, and a "
        "simple effective-sample-size estimate from autocorrelations."
    )

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
    ax1.set_title(f"Random-Walk Path, Proposal Step {main_step}")
    ax1.legend(loc="upper left")
    report.add_figure(
        "figures/mh-walk.png",
        "Metropolis-Hastings walk over target-density contours",
        fig1,
        description=(
            f"With proposal step {main_step}, the chain explores both modes while still "
            f"accepting {main_acceptance:.1%} of proposed moves."
        ),
    )

    fig2, axes2 = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    for dim, ax in enumerate(axes2):
        ax.plot(main_draws[:, dim], color="tab:blue", linewidth=0.8)
        ax.axhline(TRUE_MEAN[dim], color="black", linestyle=":", linewidth=1.2, label="Target mean")
        ax.axvline(burn, color="crimson", linestyle="--", linewidth=1.0, label="Burn-in")
        ax.set_ylabel(rf"$\theta_{dim + 1}$")
        ax.legend(loc="upper right")
    axes2[-1].set_xlabel("Draw")
    axes2[0].set_title("Trace Plots")
    report.add_figure(
        "figures/trace-plots.png",
        "Trace plots for the middle-step random-walk chain",
        fig2,
        description=(
            "Trace plots reveal whether the chain has left its starting point, whether it moves "
            "between modes, and whether the retained draws are still highly persistent."
        ),
    )

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
    axes3[0].set_title("Running Mean")
    axes3[1].axhline(0.0, color="black", linewidth=0.8)
    axes3[1].set_xlabel("Lag")
    axes3[1].set_ylabel("Autocorrelation")
    axes3[1].set_title("Autocorrelation")
    axes3[1].legend()
    fig3.tight_layout()
    report.add_figure(
        "figures/tuning-diagnostics.png",
        "Proposal tuning changes bias and persistence",
        fig3,
        description=(
            "A tiny proposal can have a comfortable acceptance rate but still move too slowly. "
            "A large proposal can switch modes, but rejection creates persistence. The best "
            "choice is empirical and target-specific."
        ),
    )

    report.add_table(
        "tables/proposal-comparison.csv",
        "Proposal-scale diagnostics",
        summary,
        description=(
            "The true mixture mean is zero. The true marginal variance of each coordinate is "
            f"{TRUE_COV[0, 0]:.2f}, which is much larger than the within-component variance "
            "because the two modes are far apart."
        ),
    )

    report.add_results(
        f"The middle proposal step, {main_step}, is used in the path and trace plots; it gives "
        f"acceptance {main_acceptance:.1%} and visible movement between modes. The small proposal "
        "accepts more often but has more persistent draws. The largest proposal is useful for "
        "jumping across the low-density middle region, but many jumps are rejected."
    )

    report.add_takeaway(
        "Metropolis-Hastings is easy to implement, but not automatic. Acceptance rates, trace "
        "plots, cumulative means, mode switching, and autocorrelation diagnose different failure "
        "modes. The key lesson is general: a sampler can be correct in theory and still be weak "
        "for a finite computation if it explores the target too slowly."
    )

    report.add_references(
        [
            "[Metropolis, N. et al. (1953). Equation of State Calculations by Fast Computing Machines. *Journal of Chemical Physics*, 21(6), 1087-1092.](https://doi.org/10.1063/1.1699114)",
            "[Hastings, W. K. (1970). Monte Carlo Sampling Methods Using Markov Chains and Their Applications. *Biometrika*, 57(1), 97-109.](https://doi.org/10.1093/biomet/57.1.97)",
            "[Chib, S. and Greenberg, E. (1995). Understanding the Metropolis-Hastings Algorithm. *The American Statistician*, 49(4), 327-335.](https://doi.org/10.1080/00031305.1995.10476177)",
        ]
    )

    report.write("README.md")


if __name__ == "__main__":
    main()
