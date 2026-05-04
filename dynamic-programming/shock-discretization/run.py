#!/usr/bin/env python3
"""Shock Discretization: Tauchen, Rouwenhorst, and Discrete Normal Grids.

Compares standard finite-state approximations to continuous shocks. The focus is
on the economic modeling choice faced by many dynamic programs: how the finite
shock process changes continuation values, policy curvature, and equilibrium
objects.

Reference: Tauchen (1986); Rouwenhorst (1995); Kopecky and Suen (2010).
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.discretize import discrete_normal, rouwenhorst, tauchen
from lib.output import ModelReport
from lib.plotting import setup_style


def stationary_distribution(P: np.ndarray, tol: float = 1e-14, max_iter: int = 50_000) -> np.ndarray:
    """Return the invariant distribution of a finite Markov chain."""
    pi = np.ones(P.shape[0]) / P.shape[0]
    for _ in range(max_iter):
        pi_next = pi @ P
        if np.max(np.abs(pi_next - pi)) < tol:
            return pi_next / pi_next.sum()
        pi = pi_next
    return pi / pi.sum()


def markov_moments(grid: np.ndarray, P: np.ndarray) -> dict[str, float]:
    """Compute unconditional moments implied by a discretized AR(1)."""
    pi = stationary_distribution(P)
    mean = float(pi @ grid)
    centered = grid - mean
    variance = float(pi @ centered**2)
    covariance = float(np.sum(pi[:, None] * P * centered[:, None] * centered[None, :]))
    return {
        "mean": mean,
        "std": float(np.sqrt(max(variance, 0.0))),
        "rho": covariance / variance if variance > 0 else np.nan,
    }


def simulate_ar1(rho: float, sigma_eps: float, T: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the continuous AR(1) from its stationary distribution."""
    rng = np.random.default_rng(seed)
    true_std = sigma_eps / np.sqrt(1.0 - rho**2)
    path = np.empty(T)
    path[0] = rng.normal(0.0, true_std)
    innovations = rng.normal(size=T - 1)
    for t in range(1, T):
        path[t] = rho * path[t - 1] + sigma_eps * innovations[t - 1]
    return path, innovations


def simulate_chain_from_uniforms(P: np.ndarray, grid: np.ndarray, z0: float, uniforms: np.ndarray) -> np.ndarray:
    """Simulate a finite chain using common random numbers."""
    idx = np.empty(len(uniforms) + 1, dtype=int)
    idx[0] = int(np.argmin(np.abs(grid - z0)))
    for t, u in enumerate(uniforms, start=1):
        cdf = np.cumsum(P[idx[t - 1]])
        idx[t] = min(int(np.searchsorted(cdf, u, side="right")), len(grid) - 1)
    return grid[idx]


def discretization_table(rho: float, sigma_eps: float, n_values: list[int]) -> pd.DataFrame:
    """Compare moment accuracy across grid sizes."""
    true_std = sigma_eps / np.sqrt(1.0 - rho**2)
    rows = []

    for n in n_values:
        z_tau, P_tau = tauchen(rho=rho, sigma=sigma_eps, n=n, m=3.0)
        z_rou, P_rou, _ = rouwenhorst(n=n, mu=0.0, sigma=sigma_eps, rho=rho)

        for method, grid, P in [
            ("Tauchen", np.asarray(z_tau), np.asarray(P_tau)),
            ("Rouwenhorst", np.asarray(z_rou).ravel(), np.asarray(P_rou)),
        ]:
            moments = markov_moments(grid, P)
            rows.append({
                "Method": method,
                "States": n,
                "Std": moments["std"],
                "Std error": moments["std"] - true_std,
                "Persistence": moments["rho"],
                "Persistence error": moments["rho"] - rho,
            })

    return pd.DataFrame(rows)


def main() -> None:
    rho = 0.95
    sigma_eps = 0.02
    n_grid = 7
    n_values = [3, 5, 7, 9, 15]
    T_sim = 180

    true_std = sigma_eps / np.sqrt(1.0 - rho**2)
    z_tau, P_tau = tauchen(rho=rho, sigma=sigma_eps, n=n_grid, m=3.0)
    z_rou, P_rou, _ = rouwenhorst(n=n_grid, mu=0.0, sigma=sigma_eps, rho=rho)
    normal_error, z_norm, p_norm = discrete_normal(n_grid, 0.0, true_std, width=3.0)

    z_tau = np.asarray(z_tau)
    P_tau = np.asarray(P_tau)
    z_rou = np.asarray(z_rou).ravel()
    P_rou = np.asarray(P_rou)
    z_norm = np.asarray(z_norm).ravel()
    p_norm = np.asarray(p_norm).ravel()

    tau_moments = markov_moments(z_tau, P_tau)
    rou_moments = markov_moments(z_rou, P_rou)
    comparison = discretization_table(rho, sigma_eps, n_values)

    print("Shock discretization comparison")
    print(f"  True AR(1): rho={rho:.2f}, innovation std={sigma_eps:.3f}, unconditional std={true_std:.4f}")
    print(f"  Tauchen std={tau_moments['std']:.4f}, rho={tau_moments['rho']:.4f}")
    print(f"  Rouwenhorst std={rou_moments['std']:.4f}, rho={rou_moments['rho']:.4f}")

    setup_style()
    report = ModelReport(
        "Discretizing Persistent Shocks",
        "Finite Markov approximations to persistent productivity and income risk.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Persistent productivity or income risk usually enters a Bellman equation through "
        "a finite Markov chain. That chain is not just a numerical convenience. Its grid "
        "points and transition probabilities determine continuation values, so they can "
        "change precautionary behavior, asset prices, equilibrium distributions, and the "
        "curvature of policy functions.\n\n"
        "This tutorial compares Tauchen, Rouwenhorst, and a discrete-normal quadrature. "
        "The point is to see how one target AR(1) can lead to different grids, transition "
        "matrices, and stationary distributions. "
        "The same issue appears downstream in the consumption-savings, RBC, and Aiyagari "
        "tutorials, where the shock process feeds directly into household choices or "
        "equilibrium objects."
    )

    report.add_equations(
        rf"""
Let $z_t$ denote the continuous shock state. The target process is the AR(1)

$$z_{{t+1}} = \rho z_t + \sigma_\epsilon \epsilon_{{t+1}}, \qquad
\epsilon_{{t+1}} \sim N(0,1).$$

Here $\rho$ is persistence and $\sigma_\epsilon$ is the innovation standard
deviation. The continuous process has unconditional variance

$$\operatorname{{Var}}(z_t) = \frac{{\sigma_\epsilon^2}}{{1-\rho^2}}.$$

A finite-state approximation replaces the continuous state with grid points
$z_1,\ldots,z_N$ and transition probabilities

$$P_{{ij}} = \Pr(z_{{t+1}} = z_j \mid z_t = z_i).$$

If a Bellman equation contains expected continuation value, the approximation
turns the integral over next period's shock into

$$\mathbb{{E}}[V(x_{{t+1}},z_{{t+1}})\mid z_t=z_i]
  \approx \sum_{{j=1}}^N P_{{ij}} V(x_{{t+1}},z_j).$$

The invariant distribution $\pi$ of the finite chain satisfies
$\pi = \pi P$ and $\sum_i \pi_i = 1$. With $\rho={rho}$ and
$\sigma_\epsilon={sigma_eps}$, the target unconditional standard deviation is
**{true_std:.4f}**.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\rho$ | {rho} | AR(1) persistence |\n"
        f"| $\\sigma_\\epsilon$ | {sigma_eps} | Innovation standard deviation |\n"
        f"| States | {n_grid} | Main comparison grid size |\n"
        f"| Grid sizes tested | {n_values} | Moment-accuracy sweep |\n"
        f"| Simulation periods | {T_sim} | Path-comparison horizon |\n"
        f"| Path benchmark | Continuous AR(1) | Ground-truth history for transition plot |"
    )

    report.add_solution_method(
        f"The pseudocode below uses the main comparison values "
        f"$\\rho={rho}$, $\\sigma_\\epsilon={sigma_eps}$, and $N={n_grid}$.\n\n"
        "**Tauchen** is transparent and interval-based. It chooses an evenly spaced grid "
        "over a fixed number of unconditional standard deviations, then assigns transition "
        "probabilities by integrating normal mass between grid-cell cutoffs.\n\n"
        "```text\n"
        "Algorithm: Tauchen AR(1) discretization\n"
        "Input: rho=0.95, sigma_epsilon=0.02, N=7, width m=3\n"
        "Output: grid z_1,...,z_N, transition matrix P, invariant distribution pi\n"
        "sigma_z = sigma_epsilon / sqrt(1 - rho^2)\n"
        "Delta = 2 * m * sigma_z / (N - 1)\n"
        "z_j = -m * sigma_z + (j - 1) * Delta,  j=1,...,N\n"
        "c_1 = -infinity, c_{N+1} = +infinity\n"
        "c_j = (z_{j-1} + z_j) / 2,  j=2,...,N\n"
        "for current state i=1,...,N:\n"
        "    for next state j=1,...,N:\n"
        "        P_ij = Phi((c_{j+1} - rho * z_i) / sigma_epsilon)\n"
        "               - Phi((c_j - rho * z_i) / sigma_epsilon)\n"
        "pi solves pi = pi P and sum_j pi_j = 1\n"
        "```\n\n"
        "**Rouwenhorst** is built for persistent processes on coarse grids. The recursive "
        "construction preserves the target autocorrelation especially well when $\\rho$ is "
        "close to one.\n\n"
        "```text\n"
        "Algorithm: Rouwenhorst AR(1) discretization\n"
        "Input: rho=0.95, sigma_epsilon=0.02, N=7\n"
        "Output: grid z_1,...,z_N, transition matrix P_N, invariant distribution pi\n"
        "p = (1 + rho) / 2\n"
        "P_2 = [[p, 1 - p], [1 - p, p]]\n"
        "for n = 3,...,N:\n"
        "    P_n = p       * upper_left(P_{n-1})\n"
        "        + (1 - p) * upper_right(P_{n-1})\n"
        "        + (1 - p) * lower_left(P_{n-1})\n"
        "        + p       * lower_right(P_{n-1})\n"
        "row-normalize P_N so each row sums to one\n"
        "sigma_z = sigma_epsilon / sqrt(1 - rho^2)\n"
        "z_j = sigma_z * sqrt(N - 1) * (2*(j - 1)/(N - 1) - 1)\n"
        "pi solves pi = pi P_N and sum_j pi_j = 1\n"
        "```\n\n"
        "**Discrete normal** approximates an IID normal random variable directly. It is "
        "useful for quadrature and independent shocks, but it should not be confused with "
        "a Markov approximation to a persistent AR(1).\n\n"
        "```text\n"
        "Algorithm: discrete-normal quadrature\n"
        "Input: mu=0, sigma_z=sigma_epsilon/sqrt(1-rho^2), N=7, width m=3\n"
        "Output: grid z_1,...,z_N and one-period probabilities p_1,...,p_N\n"
        "z_j = evenly spaced points from mu - m*sigma_z to mu + m*sigma_z\n"
        "d_j = (z_j + z_{j+1}) / 2,  j=1,...,N-1\n"
        "p_1 = Phi((d_1 - mu) / sigma_z)\n"
        "for j=2,...,N-1:\n"
        "    p_j = Phi((d_j - mu) / sigma_z) - Phi((d_{j-1} - mu) / sigma_z)\n"
        "p_N = 1 - sum_{j=1}^{N-1} p_j\n"
        "return grid z and probabilities p; there is no transition matrix\n"
        "```"
    )

    # Figure 1: stationary distributions
    fig1, ax1 = plt.subplots()
    ax1.vlines(z_tau, 0, stationary_distribution(P_tau), color="tab:blue", linewidth=3, label="Tauchen")
    ax1.scatter(z_tau, stationary_distribution(P_tau), color="tab:blue", s=45)
    ax1.vlines(z_rou, 0, stationary_distribution(P_rou), color="tab:orange", linewidth=2, label="Rouwenhorst")
    ax1.scatter(z_rou, stationary_distribution(P_rou), color="tab:orange", s=35)
    ax1.vlines(z_norm, 0, p_norm, color="tab:green", linewidth=1.5, label="Discrete normal")
    ax1.set_xlabel("Shock state")
    ax1.set_ylabel("Stationary probability")
    ax1.set_title("Stationary Mass by Approximation")
    ax1.legend()
    report.add_results(
        "Start with the stationary probabilities. They show where the finite chain spends "
        "time in the long run. Even when methods target the same continuous process, they can "
        "place mass on different support points, which changes the states used inside "
        "continuation-value calculations."
    )
    report.add_figure(
        "figures/stationary-mass.png",
        "Stationary mass across discretization methods",
        fig1,
        description=(
            "Rouwenhorst places more mass near the center while preserving the target "
            "variance. Tauchen uses the wider evenly spaced support implied by the chosen "
            "width. The discrete-normal grid matches a one-period normal distribution rather "
            "than a persistent transition law."
        ),
    )

    # Figure 2: moment accuracy
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11, 4.5))
    for method, group in comparison.groupby("Method"):
        ax2a.plot(group["States"], group["Std error"], marker="o", label=method)
        ax2b.plot(group["States"], group["Persistence error"], marker="o", label=method)
    ax2a.axhline(0.0, color="black", linewidth=0.8)
    ax2b.axhline(0.0, color="black", linewidth=0.8)
    ax2a.set_xlabel("Number of states")
    ax2a.set_ylabel("Std error")
    ax2a.set_title("Unconditional Variance")
    ax2b.set_xlabel("Number of states")
    ax2b.set_ylabel("Persistence error")
    ax2b.set_title("Autocorrelation")
    ax2b.legend()
    fig2.tight_layout()
    report.add_results(
        "The moment errors separate two questions: whether the finite chain has the right "
        "unconditional dispersion, and whether it carries shocks forward at the right rate. "
        "For persistent income or productivity risk, autocorrelation errors are often the "
        "more damaging error because they directly enter expected continuation values."
    )
    report.add_figure(
        "figures/moment-accuracy.png",
        "Coarse Tauchen grids can overstate persistence",
        fig2,
        description=(
            "For highly persistent shocks, Tauchen can distort persistence on coarse grids. "
            "Rouwenhorst is designed to keep the autocorrelation close to the target even "
            "with few states."
        ),
    )

    # Figure 3: sample paths
    true_path, true_innovations = simulate_ar1(rho, sigma_eps, T_sim, seed=123)
    common_uniforms = norm.cdf(true_innovations)
    tau_path = simulate_chain_from_uniforms(P_tau, z_tau, true_path[0], common_uniforms)
    rou_path = simulate_chain_from_uniforms(P_rou, z_rou, true_path[0], common_uniforms)
    fig3, ax3 = plt.subplots(figsize=(9, 4.5))
    ax3.plot(true_path, label="Ground truth AR(1)", color="black", linewidth=2.0, alpha=0.8)
    ax3.plot(tau_path, label="Tauchen", alpha=0.85)
    ax3.plot(rou_path, label="Rouwenhorst", alpha=0.85)
    ax3.set_xlabel("Period")
    ax3.set_ylabel("Shock state")
    ax3.set_title("Transition Histories Against the Continuous AR(1)")
    ax3.legend()
    report.add_results(
        "Sample paths make the transition matrix concrete. The black line is the continuous "
        "AR(1) ground truth. The finite chains start from the nearest grid point and use the "
        "same sequence of shock ranks, so the comparison shows how a coarse Markov chain "
        "turns a continuous history into discrete transitions."
    )
    report.add_figure(
        "figures/simulated-paths.png",
        "Transition histories against the continuous AR(1)",
        fig3,
        description=(
            "The finite paths should not match the continuous path point by point. They show "
            "which movements the solved finite model can represent when the ground-truth "
            "process is compressed to seven states."
        ),
    )

    table = comparison.copy()
    for col in ["Std", "Std error", "Persistence", "Persistence error"]:
        table[col] = table[col].map(lambda x: f"{x:.5f}")
    report.add_table(
        "tables/moment-comparison.csv",
        "Moment accuracy across discretization methods",
        table,
        description=(
            "The table reports the moments implied by each finite Markov chain. The "
            f"7-state Tauchen chain implies persistence {tau_moments['rho']:.4f}, while "
            f"the 7-state Rouwenhorst chain implies persistence {rou_moments['rho']:.4f}. "
            f"The discrete-normal standard-deviation error is {normal_error:.4e}."
        ),
    )

    report.add_results(
        "These differences matter in models such as the consumption-savings, RBC, and "
        "Aiyagari examples. In each case, the shock process is inside an expectation "
        "operator, so the Markov chain affects choices before any simulation is run."
    )

    report.add_takeaway(
        "The Markov chain is part of the economic model, not preprocessing. Tauchen is "
        "transparent and often adequate for moderate persistence. Rouwenhorst is safer for "
        "persistent income or productivity shocks on small grids because it preserves "
        "autocorrelation. Discrete-normal grids are useful for IID quadrature, but they do "
        "not replace a transition matrix when persistence matters."
    )

    report.add_references([
        "[Tauchen, G. (1986). Finite State Markov-Chain Approximations to Univariate and Vector Autoregressions. *Economics Letters*, 20(2), 177-181.](https://doi.org/10.1016/0165-1765%2886%2990168-0)",
        "[Rouwenhorst, K. G. (1995). Asset Pricing Implications of Equilibrium Business Cycle Models. In T. Cooley (ed.), *Frontiers of Business Cycle Research*. Princeton University Press.](https://doi.org/10.1515/9780691218052-014)",
        "[Kopecky, K. A. and Suen, R. M. H. (2010). Finite State Markov-Chain Approximations to Highly Persistent Processes. *Review of Economic Dynamics*, 13(3), 701-714.](https://doi.org/10.1016/j.red.2009.07.002)",
    ])
    report.write()


if __name__ == "__main__":
    main()
