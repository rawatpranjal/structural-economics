#!/usr/bin/env python3
"""Discretizing persistent income and productivity shocks.

Compares standard finite-state approximations to a Gaussian AR(1). The output
is the Markov chain that enters Bellman expectations in household and macro
models, so its variance and persistence affect policies and stationary
distributions rather than only numerical accuracy.

References: Tauchen (1986); Rouwenhorst (1995); Kopecky and Suen (2010).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.discretize import rouwenhorst, tauchen
from lib.output import ModelReport
from lib.plotting import setup_style


def stationary_distribution(P: np.ndarray, tol: float = 1e-14, max_iter: int = 50_000) -> np.ndarray:
    """Invariant distribution of a finite Markov chain via power iteration."""
    pi = np.ones(P.shape[0]) / P.shape[0]
    for _ in range(max_iter):
        pi_next = pi @ P
        if np.max(np.abs(pi_next - pi)) < tol:
            return pi_next / pi_next.sum()
        pi = pi_next
    return pi / pi.sum()


def markov_moments(grid: np.ndarray, P: np.ndarray) -> dict[str, float]:
    """Unconditional mean, std, and one-period autocorrelation of a finite chain."""
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
    """Simulate the continuous AR(1) starting from its stationary distribution."""
    rng = np.random.default_rng(seed)
    true_std = sigma_eps / np.sqrt(1.0 - rho**2)
    path = np.empty(T)
    path[0] = rng.normal(0.0, true_std)
    innovations = rng.normal(size=T - 1)
    for t in range(1, T):
        path[t] = rho * path[t - 1] + sigma_eps * innovations[t - 1]
    return path, innovations


def simulate_chain_from_uniforms(
    P: np.ndarray, grid: np.ndarray, z0: float, uniforms: np.ndarray
) -> np.ndarray:
    """Simulate a finite chain using a fixed uniform sequence (common random numbers)."""
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
    half_life = float(np.log(2.0) / -np.log(rho))

    z_tau, P_tau = tauchen(rho=rho, sigma=sigma_eps, n=n_grid, m=3.0)
    z_rou, P_rou, _ = rouwenhorst(n=n_grid, mu=0.0, sigma=sigma_eps, rho=rho)
    z_tau = np.asarray(z_tau)
    P_tau = np.asarray(P_tau)
    z_rou = np.asarray(z_rou).ravel()
    P_rou = np.asarray(P_rou)

    pi_tau = stationary_distribution(P_tau)
    pi_rou = stationary_distribution(P_rou)

    tau_moments = markov_moments(z_tau, P_tau)
    rou_moments = markov_moments(z_rou, P_rou)
    comparison = discretization_table(rho, sigma_eps, n_values)

    print("Shock discretization comparison")
    print(f"  Target AR(1): rho={rho:.2f}, sigma_eps={sigma_eps:.3f}, sigma_z={true_std:.4f}")
    print(f"  Tauchen     N=7: std={tau_moments['std']:.4f}, rho={tau_moments['rho']:.4f}")
    print(f"  Rouwenhorst N=7: std={rou_moments['std']:.4f}, rho={rou_moments['rho']:.4f}")

    setup_style()
    report = ModelReport(
        "Discretizing Persistent Shocks",
        "Why persistent shocks need a finite-state approximation before they enter Bellman equations.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Suppose a household receives low income this year. Saving depends on "
        "whether low income is likely to persist. The same issue appears when "
        "productivity shocks guide investment.\n\n"
        "The object is a persistent log income or productivity shock. We model "
        "it as a Gaussian AR(1) with persistence $\\rho$ and innovation scale "
        "$\\sigma_\\epsilon$.\n\n"
        "A Bellman equation needs finite shock states. It also needs a transition "
        "matrix for next-period expectations. The tutorial compares Tauchen and "
        "Rouwenhorst by their variance and persistence errors."
    )

    report.add_equations(
        rf"""
A household with assets $a$ faces shock $z_i$. It chooses next assets $a'$.
The continuation value averages over next-period shock states:

$$V(a,z_i) = \max_{{a' \in \mathcal{{A}}}}
[ u(Ra+\exp(z_i)-a') + \beta \sum_{{j=1}}^N P_{{ij}} V(a',z_j) ].$$

Here $R$ is the gross return factor, $\beta \in (0,1)$ is the discount factor,
and $u(\cdot)$ is a concave increasing utility function. $\mathcal{{A}}$ is the
feasible asset set.

The finite object is the grid $\{{z_1,\dots,z_N\}}$ and transition matrix $P$.
The continuous target is the Gaussian AR(1)

$$z_{{t+1}} = \rho\, z_t + \sigma_\epsilon\, \varepsilon_{{t+1}},
\qquad \varepsilon_{{t+1}} \sim \mathcal{{N}}(0,1).$$

The AR(1) has unconditional law $z_t \sim \mathcal{{N}}(0,\sigma_z^2)$ with

$$\sigma_z^2 = \frac{{\sigma_\epsilon^2}}{{1-\rho^2}},
\qquad \rho_k \equiv \mathrm{{Corr}}(z_t, z_{{t+k}}) = \rho^k.$$

For $\rho={rho}$ and $\sigma_\epsilon={sigma_eps}$, the standard deviation is
$\sigma_z = {true_std:.4f}$. The shock half-life is
$\ln 2 / (-\ln \rho) \approx {half_life:.0f}$ periods.

A finite chain replaces the conditional Gaussian law with
$P\in\mathbb{{R}}^{{N\times N}}$. Each row gives probabilities
$P_{{ij}}=\Pr(z_{{t+1}}=z_j\mid z_t=z_i)$. The conditional expectation becomes

$$\mathbb{{E}}[V(a',z_{{t+1}})\mid z_t=z_i]
= \sum_{{j=1}}^N P_{{ij}} V(a', z_j).$$

The chain has an invariant distribution $\pi$ satisfying $\pi=\pi P$ and
$\sum_i \pi_i = 1$. Two diagnostics matter:

1. Does the chain match $\sigma_z$?
2. Does the chain match $\rho$?

Variance controls risk exposure. Persistence controls expected continuation
values after good and bad shocks.
"""
    )

    report.add_model_setup(
        "The calibration is a small annual log-income or log-productivity process. "
        "It is designed for dynamic programming, not forecasting.\n\n"
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\rho$ | {rho} | AR(1) persistence |\n"
        f"| $\\sigma_\\epsilon$ | {sigma_eps} | Innovation standard deviation |\n"
        f"| $\\sigma_z$ | {true_std:.4f} | Implied unconditional standard deviation |\n"
        f"| $N$ | {n_grid} | Main comparison grid size |\n"
        f"| Grid sweep | {n_values} | Grid sizes for moment checks |\n"
        f"| Tauchen half-width $m$ | 3 | Grid bound in unconditional std deviations |\n"
        f"| $T_{{sim}}$ | {T_sim} | Simulation horizon |"
    )

    report.add_solution_method(
        "Choose a small grid and transition matrix that keep variance and "
        "persistence. Tauchen and Rouwenhorst make different compromises.\n\n"
        "### Tauchen (1986): integrate Gaussian mass between cell midpoints\n\n"
        "Tauchen places an evenly spaced grid over $[-m\\sigma_z,\\, m\\sigma_z]$. "
        "For each $z_i$, $z_{t+1}$ is normal with mean $\\rho z_i$. "
        "$P_{ij}$ is the conditional mass assigned to the cell around $z_j$, "
        "computed using the standard normal CDF $\\Phi(\\cdot)$. "
        "Endpoint cells collect remaining tail mass.\n\n"
        "```text\n"
        "Algorithm 1: Tauchen\n"
        "Input:  rho, sigma_eps, N, half-width m\n"
        "Output: grid {z_j}, transition P, invariant pi\n"
        "  sigma_z = sigma_eps / sqrt(1 - rho^2)\n"
        "  z_j     = -m*sigma_z + (j-1) * 2*m*sigma_z/(N-1)         for j = 1..N\n"
        "  c_j     = (z_{j-1} + z_j) / 2                             (cell midpoints)\n"
        "  c_1     = -inf,   c_{N+1} = +inf\n"
        "  for i = 1..N, j = 1..N:\n"
        "      P[i,j] = Phi((c_{j+1} - rho*z_i) / sigma_eps)\n"
        "             - Phi((c_j     - rho*z_i) / sigma_eps)\n"
        "  solve pi = pi P,   sum_j pi_j = 1\n"
        "```\n\n"
        "The benefit is transparency. The grid support is visible. The cost "
        "appears with high $\\rho$ and small $N$. Mass from near-tail states "
        "spills past endpoints. The last cell absorbs that mass and becomes "
        "too sticky. A wider support protects tails. A narrower support "
        "protects the center. Neither choice fixes a coarse grid.\n\n"
        "### Rouwenhorst (1995): match the moments by construction\n\n"
        "Rouwenhorst builds $P_N$ from a two-state base. The base uses "
        "$p=(1+\\rho)/2$. The recursion preserves autocorrelation as states "
        "are added. The grid is scaled to match $\\sigma_z^2$.\n\n"
        "By construction, the chain matches $\\rho$ and $\\sigma_z^2$ for any "
        "$N \\ge 2$. It has no quadrature error in those moments. The tradeoff "
        "is distributional shape. On small grids, its invariant distribution "
        "is binomial, not Gaussian.\n\n"
        "```text\n"
        "Algorithm 2: Rouwenhorst\n"
        "Input:  rho, sigma_eps, N\n"
        "Output: grid {z_j}, transition P_N, invariant pi\n"
        "  p   = (1 + rho) / 2\n"
        "  P_2 = [[p, 1-p],\n"
        "         [1-p, p]]\n"
        "  for n = 3..N:\n"
        "      A_TL = embed P_{n-1} in top-left of n x n zero matrix\n"
        "      A_TR = embed P_{n-1} in top-right of n x n zero matrix\n"
        "      A_BL = embed P_{n-1} in bottom-left of n x n zero matrix\n"
        "      A_BR = embed P_{n-1} in bottom-right of n x n zero matrix\n"
        "      P_n  = p*A_TL + (1-p)*A_TR + (1-p)*A_BL + p*A_BR\n"
        "      row-normalize interior rows of P_n     (they receive two contributions)\n"
        "  sigma_z = sigma_eps / sqrt(1 - rho^2)\n"
        "  z_j     = sigma_z * sqrt(N-1) * (2*(j-1)/(N-1) - 1)        for j = 1..N\n"
        "  pi_j    = binomial(N-1, j-1) / 2^{N-1}\n"
        "```\n\n"
        "For highly persistent shocks on coarse grids, this is usually safer. "
        "It protects the moments that enter continuation values."
    )

    # --------------------------------------------------------------------
    # Figure 1: stationary mass with Gaussian density ground-truth overlay
    # --------------------------------------------------------------------
    fig1, ax1 = plt.subplots()
    z_dense = np.linspace(-3.5 * true_std, 3.5 * true_std, 400)
    tau_step = float(z_tau[1] - z_tau[0])
    pdf_dense = norm.pdf(z_dense, loc=0.0, scale=true_std) * tau_step
    ax1.plot(
        z_dense,
        pdf_dense,
        color="black",
        linestyle="--",
        linewidth=1.4,
        alpha=0.7,
        label=r"Ground truth $\mathcal{N}(0,\sigma_z^2)\times\Delta_{\mathrm{Tau}}$",
    )
    ax1.vlines(z_tau, 0, pi_tau, color="tab:blue", linewidth=3, label="Tauchen")
    ax1.scatter(z_tau, pi_tau, color="tab:blue", s=45)
    ax1.vlines(z_rou, 0, pi_rou, color="tab:orange", linewidth=2, label="Rouwenhorst")
    ax1.scatter(z_rou, pi_rou, color="tab:orange", s=35)
    ax1.set_xlabel("Shock state $z$")
    ax1.set_ylabel("Probability mass")
    ax1.set_title("Where the Chain Lives")
    ax1.legend(loc="upper right", fontsize=9)
    report.add_results(
        "Stationary mass shows where each chain puts probability. The dashed "
        "curve is the AR(1)'s long-run Gaussian density, scaled by the Tauchen "
        "cell width. Tauchen follows the curve near the center, but it puts "
        "extra mass in outer states. That extra tail mass creates the variance "
        "error. Rouwenhorst has a binomial invariant distribution, so its center "
        "is heavier and its tails are thinner."
    )
    report.add_figure(
        "figures/stationary-mass.png",
        "Stationary mass for Tauchen and Rouwenhorst",
        fig1,
    )

    # --------------------------------------------------------------------
    # Figure 2: moment errors as the grid refines
    # --------------------------------------------------------------------
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11, 4.5))
    for method, group in comparison.groupby("Method"):
        ax2a.plot(group["States"], group["Std error"], marker="o", label=method)
        ax2b.plot(group["States"], group["Persistence error"], marker="o", label=method)
    ax2a.axhline(0.0, color="black", linewidth=0.8)
    ax2b.axhline(0.0, color="black", linewidth=0.8)
    ax2a.set_xlabel("Number of states $N$")
    ax2a.set_ylabel(r"$\hat\sigma_z - \sigma_z$")
    ax2a.set_title("Unconditional std error")
    ax2b.set_xlabel("Number of states $N$")
    ax2b.set_ylabel(r"$\hat\rho - \rho$")
    ax2b.set_title("Persistence error")
    ax2b.legend()
    fig2.tight_layout()
    report.add_results(
        "Moment errors show the main diagnostic. The zero line is the AR(1) "
        "target. Rouwenhorst stays on zero because the recursion enforces "
        "variance and persistence. Tauchen approaches the targets as $N$ "
        "grows. With $N=3$, Tauchen is almost absorbing, so persistence is "
        "near one. At $\\rho=0.95$, small persistence errors affect each "
        "continuation value."
    )
    report.add_figure(
        "figures/moment-accuracy.png",
        "Moment errors by grid size",
        fig2,
    )

    # --------------------------------------------------------------------
    # Figure 3: simulated path against the continuous AR(1)
    # --------------------------------------------------------------------
    true_path, true_innovations = simulate_ar1(rho, sigma_eps, T_sim, seed=123)
    common_uniforms = norm.cdf(true_innovations)
    tau_path = simulate_chain_from_uniforms(P_tau, z_tau, true_path[0], common_uniforms)
    rou_path = simulate_chain_from_uniforms(P_rou, z_rou, true_path[0], common_uniforms)
    fig3, ax3 = plt.subplots(figsize=(9, 4.5))
    ax3.plot(true_path, label="Continuous AR(1) ground truth", color="black", linewidth=2.0, alpha=0.85)
    ax3.plot(tau_path, label=r"Tauchen ($N=7$)", alpha=0.85)
    ax3.plot(rou_path, label=r"Rouwenhorst ($N=7$)", alpha=0.85)
    ax3.set_xlabel("Period $t$")
    ax3.set_ylabel("Shock state $z_t$")
    ax3.set_title("Transition Histories Against the Continuous AR(1)")
    ax3.legend(loc="upper right", fontsize=9)
    report.add_results(
        "Simulated paths make the transition matrix visible. The finite chains "
        "receive the same innovation ranks as the AR(1) path. They move on "
        "coarse grids, so they cannot match the continuous path point by "
        "point. The useful check is the rhythm of persistence. Rouwenhorst "
        "tracks slow drift more cleanly at $N=7$."
    )
    report.add_figure(
        "figures/simulated-paths.png",
        "Simulated AR(1) and finite-chain paths",
        fig3,
    )

    # --------------------------------------------------------------------
    # Table: numerical detail behind the moment-accuracy figure
    # --------------------------------------------------------------------
    table = comparison.copy()
    for col in ["Std", "Std error", "Persistence", "Persistence error"]:
        table[col] = table[col].map(lambda x: f"{x:.5f}")
    report.add_table(
        "tables/moment-comparison.csv",
        "Moment accuracy across discretization methods",
        table,
        description=(
            "Numerical detail behind the moment-error figure. The 7-state Tauchen chain "
            f"implies persistence {tau_moments['rho']:.4f} against a target of {rho}. "
            "Rouwenhorst matches the target at every $N$."
        ),
    )

    report.add_takeaway(
        "Discretization is part of the economic model. With persistent shocks "
        "and small $N$, Rouwenhorst is the safer default. It matches "
        "$\\sigma_z$ and $\\rho$ by construction. Tauchen is transparent and "
        "can approximate the Gaussian shape well on finer grids. At "
        "$\\rho=0.95$ and $N=7$, Tauchen overstates persistence enough to "
        "change continuation values. Choose the chain by the moments that "
        "matter in the Bellman equation."
    )

    report.add_references([
        "[Tauchen, G. (1986). Finite State Markov-Chain Approximations to Univariate and Vector Autoregressions. *Economics Letters*, 20(2), 177-181.](https://doi.org/10.1016/0165-1765%2886%2990168-0)",
        "[Rouwenhorst, K. G. (1995). Asset Pricing Implications of Equilibrium Business Cycle Models. In T. Cooley (ed.), *Frontiers of Business Cycle Research*. Princeton University Press.](https://doi.org/10.1515/9780691218052-014)",
        "[Kopecky, K. A. and Suen, R. M. H. (2010). Finite State Markov-Chain Approximations to Highly Persistent Processes. *Review of Economic Dynamics*, 13(3), 701-714.](https://doi.org/10.1016/j.red.2010.02.002)",
    ])
    report.write()


if __name__ == "__main__":
    main()
