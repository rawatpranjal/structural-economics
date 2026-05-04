#!/usr/bin/env python3
"""Shock Discretization: Tauchen, Rouwenhorst, and Discrete Normal Grids.

Compares standard finite-state approximations to continuous shocks. The focus is
on the practical question faced by many dynamic programs: how much of the
continuous income or productivity process is preserved after discretization?

Reference: Tauchen (1986); Rouwenhorst (1995); Kopecky and Suen (2010).
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def simulate_chain(P: np.ndarray, grid: np.ndarray, T: int, seed: int) -> np.ndarray:
    """Simulate a Markov chain from its stationary distribution."""
    rng = np.random.default_rng(seed)
    pi = stationary_distribution(P)
    idx = np.empty(T, dtype=int)
    idx[0] = rng.choice(len(grid), p=pi)
    for t in range(1, T):
        idx[t] = rng.choice(len(grid), p=P[idx[t - 1]])
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
        "Shock Discretization",
        "Finite-state approximations to continuous productivity and income shocks.",
    )

    report.add_overview(
        "Many dynamic economic models begin with a continuous shock process but solve on a "
        "finite grid. This tutorial compares three workhorse approximations: Tauchen's "
        "normal grid for AR(1) processes, Rouwenhorst's highly persistent Markov chain, "
        "and a simple discrete-normal quadrature for one-period shocks.\n\n"
        "The goal is not to pick one method forever. The goal is to understand what each "
        "method preserves: unconditional variance, persistence, tail support, and transition "
        "probabilities."
    )

    report.add_equations(
        rf"""
We start from the continuous AR(1):

$$z_{{t+1}} = \rho z_t + \sigma_\epsilon \epsilon_{{t+1}}, \qquad
\epsilon_{{t+1}} \sim N(0,1).$$

The continuous process has unconditional variance:

$$\operatorname{{Var}}(z_t) = \frac{{\sigma_\epsilon^2}}{{1-\rho^2}}.$$

With $\rho={rho}$ and $\sigma_\epsilon={sigma_eps}$, the true unconditional
standard deviation is **{true_std:.4f}**.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\rho$ | {rho} | AR(1) persistence |\n"
        f"| $\\sigma_\\epsilon$ | {sigma_eps} | Innovation standard deviation |\n"
        f"| States | {n_grid} | Main comparison grid size |\n"
        f"| Grid sizes tested | {n_values} | Moment-accuracy sweep |"
    )

    report.add_solution_method(
        "**Tauchen** chooses an evenly spaced grid over a fixed number of unconditional "
        "standard deviations and assigns transition probabilities by integrating normal "
        "mass between grid-cell cutoffs.\n\n"
        "**Rouwenhorst** builds the transition matrix recursively. It is especially useful "
        "when shocks are persistent because it preserves persistence well with few states.\n\n"
        "**Discrete normal** approximates a one-period normal distribution directly. It is "
        "not a Markov approximation to persistence, but it is useful for quadrature and "
        "independent shocks."
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
    ax1.set_title("Finite-State Probability Mass")
    ax1.legend()
    report.add_figure(
        "figures/stationary-mass.png",
        "Stationary distributions implied by 7-state discretizations",
        fig1,
        description=(
            "Rouwenhorst places more mass near the center but keeps wider tail support. "
            "Tauchen uses evenly spaced cutoffs over the chosen width. The discrete-normal "
            "grid matches a one-period normal distribution rather than an AR(1) transition."
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
    ax2a.set_title("Variance Accuracy")
    ax2b.set_xlabel("Number of states")
    ax2b.set_ylabel("Persistence error")
    ax2b.set_title("Persistence Accuracy")
    ax2b.legend()
    fig2.tight_layout()
    report.add_figure(
        "figures/moment-accuracy.png",
        "Moment errors relative to the continuous AR(1)",
        fig2,
        description=(
            "For highly persistent shocks, Tauchen can miss persistence on coarse grids. "
            "Rouwenhorst is designed to preserve the persistence parameter more tightly."
        ),
    )

    # Figure 3: sample paths
    tau_path = simulate_chain(P_tau, z_tau, T_sim, seed=123)
    rou_path = simulate_chain(P_rou, z_rou, T_sim, seed=123)
    fig3, ax3 = plt.subplots(figsize=(9, 4.5))
    ax3.plot(tau_path, label="Tauchen", alpha=0.85)
    ax3.plot(rou_path, label="Rouwenhorst", alpha=0.85)
    ax3.set_xlabel("Period")
    ax3.set_ylabel("Shock state")
    ax3.set_title("Simulated Markov Chains")
    ax3.legend()
    report.add_figure(
        "figures/simulated-paths.png",
        "Sample paths from Tauchen and Rouwenhorst chains",
        fig3,
        description=(
            "The two chains can share the same target process but generate visibly different "
            "finite-state dynamics on coarse grids. This matters for policy functions near "
            "borrowing constraints or other nonlinear regions."
        ),
    )

    table = comparison.copy()
    for col in ["Std", "Std error", "Persistence", "Persistence error"]:
        table[col] = table[col].map(lambda x: f"{x:.5f}")
    report.add_table(
        "tables/moment-comparison.csv",
        "Moment accuracy across discretization methods",
        table,
        description="The table reports moments implied by the finite Markov chains.",
    )

    report.add_results(
        f"The 7-state Tauchen chain implies persistence {tau_moments['rho']:.4f}, while "
        f"the 7-state Rouwenhorst chain implies persistence {rou_moments['rho']:.4f}. "
        f"The discrete-normal standard-deviation error is {normal_error:.4e}."
    )

    report.add_takeaway(
        "Discretization is part of the model, not a harmless preprocessing step. Tauchen is "
        "transparent and often adequate for moderate persistence. Rouwenhorst is safer for "
        "persistent income or productivity shocks because it preserves autocorrelation on "
        "small grids. Discrete-normal grids are useful for independent quadrature but do not "
        "replace a Markov transition matrix when persistence matters."
    )

    report.add_references([
        "Tauchen, G. (1986). Finite state Markov-chain approximations to univariate and vector autoregressions. Economics Letters.",
        "Rouwenhorst, K. G. (1995). Asset pricing implications of equilibrium business cycle models. In Frontiers of Business Cycle Research.",
        "Kopecky, K. A. and Suen, R. M. H. (2010). Finite state Markov-chain approximations to highly persistent processes. Review of Economic Dynamics.",
    ])
    report.write()


if __name__ == "__main__":
    main()
