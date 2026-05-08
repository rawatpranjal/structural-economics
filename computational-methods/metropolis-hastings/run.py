#!/usr/bin/env python3
"""Random-walk Metropolis-Hastings for a two-regime structural posterior.

This tutorial samples from a mixture of two bivariate normals. The mixture
stands in for a structural posterior where two parameter regions fit the data,
making proposal tuning and mode switching visible in trace plots and diagnostics.
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
        "Sampling a Two-Regime Structural Posterior",
        "Random-walk Metropolis-Hastings for structural parameters with two posterior modes.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Suppose a structural model leaves two parameter regions that fit the same data. "
        "Counterfactual prices or welfare can depend on which region receives posterior weight.\n\n"
        "The object is the posterior over $\\theta=(\\theta_1,\\theta_2)$. This tutorial represents "
        "it with two Gaussian modes, which stand for two plausible structural regimes.\n\n"
        "We can evaluate the posterior density up to a constant, but averages require integration. "
        "Random-walk Metropolis-Hastings replaces the integral with draws from a Markov chain. "
        "The finite run is useful only when it crosses modes often enough."
    )

    report.add_equations(
        r"""
Let $D$ denote the data, and let $\theta=(\theta_1,\theta_2)$ collect two
structural parameters. The posterior kernel is

$$
\pi(\theta \mid D) \propto L(D \mid \theta) p_0(\theta).
$$

The target used here is a two-component mixture:

$$
\begin{aligned}
\pi(\theta \mid D)
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
= \min[1, \pi(\theta^\star \mid D) / \pi(\theta_t \mid D)].
$$

Retained draws approximate posterior averages for any counterfactual object
$g(\theta)$. The approximation is weak when the chain rarely crosses modes.
"""
    )

    report.add_model_setup(
        f"| Object | Value |\n"
        f"|--------|-------|\n"
        f"| Posterior interpretation | Two empirically plausible structural regimes |\n"
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
        "Random-walk Metropolis-Hastings needs the posterior kernel at current and proposed "
        "parameter values. The normalizing constant cancels from the acceptance ratio. The script "
        "runs three proposal scales to show the tuning tradeoff.\n\n"
        "```text\n"
        "Algorithm: random-walk Metropolis-Hastings\n"
        "Input: log posterior kernel ell(theta), proposal scale s, initial theta_0, draws T\n"
        "Output: draws from pi(theta | D), plus mode-crossing summaries\n"
        "1. Set theta = theta_0 and compute ell(theta)\n"
        "2. For t = 1, ..., T:\n"
        "       propose theta_star = theta + s * eta_t, eta_t ~ N(0, I)\n"
        "       compute log alpha = ell(theta_star) - ell(theta)\n"
        "       accept theta_star with probability min(1, exp(log alpha))\n"
        "       otherwise repeat the current theta\n"
        "3. Drop burn-in draws\n"
        "4. Report acceptance, mode switches, posterior mean error, and ESS\n"
        "```\n\n"
        "Proposal scale $s$ controls local move size. Tiny steps accept often but cross modes "
        "slowly. Large steps cross low-density regions more often, but many proposals are "
        "rejected. The known mixture mean lets the code measure finite-chain error."
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
        "Metropolis-Hastings walk over structural-posterior contours",
        fig1,
        description=(
            f"With proposal step {main_step}, the chain visits both regimes and accepts "
            f"{main_acceptance:.1%} of proposed moves."
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
            "The traces show burn-in, mode crossing, and persistence in the retained draws."
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
        "Proposal tuning changes posterior bias and persistence",
        fig3,
        description=(
            "The running mean and autocorrelation show how proposal scale changes finite-chain "
            "error."
        ),
    )

    report.add_table(
        "tables/proposal-comparison.csv",
        "Proposal-scale diagnostics",
        summary,
        description=(
            "The true posterior mean is zero. Each coordinate has marginal variance "
            f"{TRUE_COV[0, 0]:.2f} because the modes are far apart."
        ),
    )

    report.add_results(
        f"The middle proposal step, {main_step}, is used in the path and trace plots. It gives "
        f"acceptance {main_acceptance:.1%} and moves between regimes. The smallest proposal "
        "accepts most often but crosses modes slowly. The largest proposal has lower acceptance, "
        "more mode switches, and the smallest mean error. The table shows why acceptance rate "
        "alone is not enough."
    )

    report.add_takeaway(
        "Metropolis-Hastings turns a posterior kernel into draws for structural uncertainty and "
        "counterfactual averages. A finite run can still weight regimes incorrectly. Use traces, "
        "cumulative means, mode switches, and autocorrelation to check whether the chain supports "
        "the economic conclusion."
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
