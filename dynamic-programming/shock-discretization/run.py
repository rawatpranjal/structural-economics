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
from lib.discretize import discrete_normal, rouwenhorst, tauchen
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
    normal_error, z_norm, p_norm = discrete_normal(n_grid, 0.0, true_std, width=3.0)

    z_tau = np.asarray(z_tau)
    P_tau = np.asarray(P_tau)
    z_rou = np.asarray(z_rou).ravel()
    P_rou = np.asarray(P_rou)
    z_norm = np.asarray(z_norm).ravel()
    p_norm = np.asarray(p_norm).ravel()

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
        "Turning a Gaussian AR(1) into the finite Markov chain used in Bellman equations.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Start with a household or planner who faces persistent income or productivity "
        "risk. Income might be $y_t=\\exp(z_t)$ in a buffer-stock saving model, or "
        "productivity might be $A_t=\\exp(z_t)$ in an RBC planner problem. The economic "
        "question is not how to draw a prettier grid. It is how to replace the "
        "continuous shock $z_t$ with a finite Markov chain without changing the "
        "insurance motive, persistence, and long-run distribution that the dynamic "
        "model sees.\n\n"
        "This tutorial isolates that step before solving a full dynamic program. A "
        "Gaussian AR(1) is approximated by two standard finite-state methods. "
        "**Tauchen** lays a uniform grid over a few unconditional standard deviations "
        "and integrates conditional Gaussian mass between cell midpoints. "
        "**Rouwenhorst** builds the transition matrix recursively so that the chain "
        "matches the AR(1) variance and one-period autocorrelation by construction.\n\n"
        "For annual income or productivity with $\\rho \\approx 0.95$, this choice is "
        "economically meaningful. With $N=7$, Tauchen overstates persistence by about "
        "a percentage point and the unconditional standard deviation by more than "
        "20 percent, while Rouwenhorst matches both target moments. Those errors "
        "feed directly into continuation values in the downstream "
        "[consumption-savings](../consumption-savings/), "
        "[Aiyagari](../aiyagari/), and [RBC](../rbc/) tutorials."
    )

    report.add_equations(
        rf"""
A simple income-risk Bellman equation shows where the finite Markov chain
enters. If assets are $a$, current shock state is $z_i$, and next assets are
$a'$, the continuation term is a weighted sum over next-period shock states:

$$V(a,z_i) = \max_{{a' \in \mathcal{{A}}}}
[ u(Ra+\exp(z_i)-a') + \beta \sum_{{j=1}}^N P_{{ij}} V(a',z_j) ].$$

The computational object needed by this Bellman equation is the grid
$\{{z_1,\dots,z_N\}}$ and the transition matrix $P$. The continuous process
being approximated is the Gaussian AR(1)

$$z_{{t+1}} = \rho\, z_t + \sigma_\epsilon\, \varepsilon_{{t+1}},
\qquad \varepsilon_{{t+1}} \sim \mathcal{{N}}(0,1).$$

Its unconditional law is $z_t \sim \mathcal{{N}}(0, \sigma_z^2)$ with

$$\sigma_z^2 \;=\; \frac{{\sigma_\epsilon^2}}{{1-\rho^2}},
\qquad \rho_k \;\equiv\; \mathrm{{Corr}}(z_t, z_{{t+k}}) \;=\; \rho^k.$$

For $\rho={rho}$ and $\sigma_\epsilon={sigma_eps}$, the unconditional standard
deviation is
$\sigma_z = {true_std:.4f}$, with shock half-life
$\ln 2 / (-\ln \rho) \approx {half_life:.0f}$ periods.

A finite-state approximation replaces the conditional Gaussian law with a
row-stochastic matrix $P\in\mathbb{{R}}^{{N\times N}}$, where
$P_{{ij}}=\Pr(z_{{t+1}}=z_j\mid z_t=z_i)$. The conditional expectation in any
Bellman equation then becomes

$$\mathbb{{E}}[V(x_{{t+1}},z_{{t+1}})\mid z_t=z_i]
= \sum_{{j=1}}^N P_{{ij}} V(x_{{t+1}}, z_j).$$

The chain has an invariant distribution $\pi$ satisfying $\pi=\pi P$ and
$\sum_i \pi_i = 1$. Two diagnostics decide whether it is a faithful stand-in
for the AR(1):

1. Does the implied unconditional standard deviation match $\sigma_z$?
2. Does the implied one-period autocorrelation match $\rho$?

The first moment controls how much cross-sectional risk the model sees. The
second controls how much a good or bad shock changes future continuation values.
"""
    )

    report.add_model_setup(
        "The calibration is deliberately small. It should be read as an annual log "
        "income or log productivity shock that will later be embedded in a dynamic "
        "program, not as a standalone forecasting exercise.\n\n"
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\rho$ | {rho} | AR(1) persistence (annual productivity-like calibration) |\n"
        f"| $\\sigma_\\epsilon$ | {sigma_eps} | Innovation standard deviation |\n"
        f"| $\\sigma_z$ | {true_std:.4f} | Implied unconditional standard deviation |\n"
        f"| $N$ | {n_grid} | Main comparison grid size |\n"
        f"| Grid sweep | {n_values} | Grid sizes used in the moment-accuracy panel |\n"
        f"| Tauchen half-width $m$ | 3 | Grid bound in unconditional std deviations |\n"
        f"| $T_{{sim}}$ | {T_sim} | Simulation horizon for transition-history figure |"
    )

    report.add_solution_method(
        "The computational task is to choose a small state space and a transition "
        "matrix that preserve the economic features of the continuous shock process. "
        "Tauchen and Rouwenhorst solve that task in different ways.\n\n"
        "### Tauchen (1986): integrate Gaussian mass between cell midpoints\n\n"
        "Place an evenly spaced grid over $[-m\\sigma_z,\\, m\\sigma_z]$. For "
        "each starting state $z_i$ the conditional law of $z_{t+1}$ is "
        "$\\mathcal{N}(\\rho z_i,\\,\\sigma_\\epsilon^2)$, and $P_{ij}$ is the "
        "mass that conditional Gaussian assigns to the cell around $z_j$. "
        "Endpoint cells absorb the remaining tail mass.\n\n"
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
        "The appeal is transparency. The grid has a visible support choice, and "
        "refining $N$ recovers the conditional Gaussian. The cost shows up when "
        "$\\rho$ is close to one and $N$ is small. Conditional mass from a near-tail "
        "$z_i$ spills beyond the endpoint and gets absorbed by the last cell. The "
        "chain then puts too much mass in persistent tail states. Widening the "
        "support with a larger $m$ protects the tails but thins the center; shrinking "
        "$m$ protects the center but truncates rare states.\n\n"
        "### Rouwenhorst (1995): match the moments by construction\n\n"
        "Build $P_N$ recursively from a 2-state base whose persistence "
        "$p=(1+\\rho)/2$ is calibrated to deliver the AR(1) one-period "
        "autocorrelation. The grid is then chosen so that the chain has "
        "unconditional variance $\\sigma_z^2$ exactly. By construction the "
        "resulting chain matches $\\rho$ and $\\sigma_z^2$ for any $N \\ge 2$, "
        "with no quadrature error. The price is that $\\pi_j = "
        "\\binom{N-1}{j-1}/2^{N-1}$ is binomial, so on small grids the *shape* "
        "of the stationary distribution is wrong even though the first two "
        "moments are right.\n\n"
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
        "For highly persistent processes on coarse grids, this is usually the safer "
        "economic approximation. The moments that drive continuation values are exact "
        "rather than approached gradually as the grid grows.\n\n"
        "### Discrete-normal quadrature (for contrast only)\n\n"
        "The first figure also shows a discrete-normal quadrature, which "
        "approximates the *unconditional* law $\\mathcal{N}(0,\\sigma_z^2)$ "
        "directly without a transition matrix. It is the right tool for IID "
        "shocks (taste shocks, measurement error, one-shot quadrature inside "
        "another expectation) and the wrong tool for persistent shocks. It is "
        "included to make the distinction explicit: a discrete-normal grid "
        "loses all serial dependence."
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
    ax1.vlines(z_norm, 0, p_norm, color="tab:green", linewidth=1.5, label="Discrete normal (IID)")
    ax1.set_xlabel("Shock state $z$")
    ax1.set_ylabel("Probability mass")
    ax1.set_title("Where the Chain Lives")
    ax1.legend(loc="upper right", fontsize=9)
    report.add_results(
        "The first figure shows the stationary mass each chain places on its "
        "support. The dashed reference curve is the continuous AR(1)'s long-run "
        "Gaussian density rescaled by the Tauchen cell width, so a faithful "
        "Tauchen approximation should sit on the curve. Tauchen tracks it "
        "reasonably well at $N=7$ but bleeds extra mass into its outermost "
        "states, which is exactly the source of the variance overstatement "
        "documented next. Rouwenhorst sits on a different shape: its $\\pi$ is "
        "binomial, so even though $\\sigma_z$ is exact, the central states are "
        "heavier and the tails thinner than the Gaussian. The discrete-normal "
        "stems are included only to flag the contrast: without a transition "
        "matrix, that grid carries no information about persistence."
    )
    report.add_figure(
        "figures/stationary-mass.png",
        "Stationary mass across discretization methods",
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
        "The second figure separates the two diagnostics. The horizontal axis "
        "is grid size; the zero line is the true AR(1) moment in each panel. "
        "Rouwenhorst sits on zero at every $N$: the construction enforces both "
        "moments analytically. Tauchen approaches the targets only as $N$ "
        "grows, and convergence is slow when $\\rho$ is close to one. The "
        "$N=3$ Tauchen case is instructive: with only three states the chain "
        "rarely transitions, so the implied autocorrelation collapses to "
        "essentially one regardless of the target. For PhD-style "
        "consumption-savings or RBC calibrations that pair $\\rho \\approx 0.95$ "
        "with small $N$, the persistence error is the more damaging one: it "
        "enters every continuation value off the chain and biases "
        "precautionary saving and asset prices."
    )
    report.add_figure(
        "figures/moment-accuracy.png",
        "Moment errors across grid sizes",
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
        "A common-random-numbers simulation makes the transition matrix "
        "concrete. Both finite chains receive the same sequence of innovation "
        "ranks fed through their own conditional CDFs, so any deviation from "
        "the black AR(1) path comes from discretization alone. The chains "
        "step between grid points: what was a smooth path becomes a coarse "
        "staircase. The right question is not whether the staircase tracks "
        "the continuous line point by point (it cannot at $N=7$) but whether "
        "the rhythm of its movements is reasonable. Tauchen occasionally "
        "jumps to its wider tail states; Rouwenhorst is forced to stay closer "
        "to the center by its binomial $\\pi$, but reproduces the slow drift "
        "of a high-$\\rho$ process more cleanly."
    )
    report.add_figure(
        "figures/simulated-paths.png",
        "Transition histories against the continuous AR(1)",
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
            "Numerical detail behind the previous figure. The 7-state Tauchen chain implies "
            f"persistence {tau_moments['rho']:.4f} against a target of {rho}; Rouwenhorst "
            "matches the target to numerical precision at every $N$. The discrete-normal "
            f"unconditional-std error at $N=7$ is {normal_error:.2e}, reported separately "
            "because that grid carries no transition law."
        ),
    )

    report.add_takeaway(
        "The discretized shock process is part of the economic model, not "
        "preprocessing. With persistent shocks and small $N$, Rouwenhorst is "
        "the safer default: it nails $\\sigma_z$ and $\\rho$ by construction, "
        "which controls the bias entering expected continuation values in any "
        "Bellman problem the chain feeds into. Tauchen is more transparent and "
        "delivers a Gaussian-like stationary distribution, which can matter "
        "when the *shape* of the unconditional law shows up in the equilibrium "
        "object (for example, wealth distributions aggregating over income "
        "states). It is fine when persistence is moderate or the grid is fine "
        "enough. But with $\\rho=0.95$ and $N=7$, it overstates persistence by "
        "about a percentage point, enough to perturb precautionary saving and "
        "asset prices in the downstream "
        "[consumption-savings](../consumption-savings/) and "
        "[Aiyagari](../aiyagari/) tutorials. Discrete-normal grids belong to a "
        "different problem entirely: they approximate IID shocks and discard "
        "serial dependence."
    )

    report.add_references([
        "[Tauchen, G. (1986). Finite State Markov-Chain Approximations to Univariate and Vector Autoregressions. *Economics Letters*, 20(2), 177-181.](https://doi.org/10.1016/0165-1765%2886%2990168-0)",
        "[Rouwenhorst, K. G. (1995). Asset Pricing Implications of Equilibrium Business Cycle Models. In T. Cooley (ed.), *Frontiers of Business Cycle Research*. Princeton University Press.](https://doi.org/10.1515/9780691218052-014)",
        "[Kopecky, K. A. and Suen, R. M. H. (2010). Finite State Markov-Chain Approximations to Highly Persistent Processes. *Review of Economic Dynamics*, 13(3), 701-714.](https://doi.org/10.1016/j.red.2009.07.002)",
    ])
    report.write()


if __name__ == "__main__":
    main()
