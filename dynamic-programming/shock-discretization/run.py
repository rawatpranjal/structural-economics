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
from lib.plotting import save_figure, save_thumbnail, setup_style


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

    # =========================================================================
    # Figure 1 (1x2): stationary mass vs Gaussian overlay | transition heatmap
    # Left: where each chain's invariant distribution sits relative to the
    # continuous AR(1) density. Right: the Rouwenhorst N=7 transition matrix
    # as a heatmap -- each cell is P[i,j], the probability of moving from
    # state i to state j.
    # =========================================================================
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 4.8))

    z_dense = np.linspace(-3.5 * true_std, 3.5 * true_std, 400)
    tau_step = float(z_tau[1] - z_tau[0])
    pdf_dense = norm.pdf(z_dense, loc=0.0, scale=true_std) * tau_step
    ax1a.plot(
        z_dense,
        pdf_dense,
        color="black",
        linestyle="--",
        linewidth=1.4,
        alpha=0.7,
        label=r"AR(1) density $\times\,\Delta_{\mathrm{Tau}}$",
    )
    ax1a.vlines(z_tau, 0, pi_tau, color="tab:blue", linewidth=3, label="Tauchen")
    ax1a.scatter(z_tau, pi_tau, color="tab:blue", s=45)
    ax1a.vlines(z_rou, 0, pi_rou, color="tab:orange", linewidth=2, label="Rouwenhorst")
    ax1a.scatter(z_rou, pi_rou, color="tab:orange", s=35)
    ax1a.set_xlabel("Shock state $z$")
    ax1a.set_ylabel("Probability mass")
    ax1a.set_title("Invariant distribution")
    ax1a.legend(loc="upper right", fontsize=9)

    # Rouwenhorst N=7 transition matrix heatmap
    im = ax1b.imshow(P_rou, cmap="Blues", vmin=0, vmax=P_rou.max())
    fig1.colorbar(im, ax=ax1b, fraction=0.046, pad=0.04)
    state_labels = [f"$z_{{{j+1}}}$" for j in range(n_grid)]
    ax1b.set_xticks(range(n_grid))
    ax1b.set_yticks(range(n_grid))
    ax1b.set_xticklabels(state_labels, fontsize=8)
    ax1b.set_yticklabels(state_labels, fontsize=8)
    ax1b.set_xlabel("Next state $z'$")
    ax1b.set_ylabel("Current state $z$")
    ax1b.set_title(r"Rouwenhorst $P$ ($N=7$)")

    fig1.tight_layout()
    save_figure(fig1, "figures/stationary-mass.png", dpi=150)

    # =========================================================================
    # Figure 2 (2x2): moment errors (top row) | simulated paths (bottom row)
    # Top-left: unconditional std error vs grid size N.
    # Top-right: persistence error vs grid size N.
    # Bottom row: Tauchen and Rouwenhorst chains against continuous AR(1),
    #   driven by common random numbers so the paths are comparable.
    # =========================================================================
    true_path, true_innovations = simulate_ar1(rho, sigma_eps, T_sim, seed=123)
    common_uniforms = norm.cdf(true_innovations)
    tau_path = simulate_chain_from_uniforms(P_tau, z_tau, true_path[0], common_uniforms)
    rou_path = simulate_chain_from_uniforms(P_rou, z_rou, true_path[0], common_uniforms)

    fig2, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    ax2a, ax2b = axes[0]
    ax2c, ax2d = axes[1]

    for method, group in comparison.groupby("Method"):
        ax2a.plot(group["States"], group["Std error"], marker="o", label=method)
        ax2b.plot(group["States"], group["Persistence error"], marker="o", label=method)
    ax2a.axhline(0.0, color="black", linewidth=0.8)
    ax2b.axhline(0.0, color="black", linewidth=0.8)
    ax2a.set_xlabel("Number of states $N$")
    ax2a.set_ylabel(r"$\hat\sigma_z - \sigma_z$")
    ax2a.set_title("Unconditional std error")
    ax2a.legend()
    ax2b.set_xlabel("Number of states $N$")
    ax2b.set_ylabel(r"$\hat\rho - \rho$")
    ax2b.set_title("Persistence error")
    ax2b.legend()

    # Bottom-left: Tauchen path vs continuous AR(1)
    t_axis = np.arange(T_sim)
    ax2c.plot(t_axis, true_path, color="black", linewidth=1.8, alpha=0.75, label="Continuous AR(1)")
    ax2c.step(t_axis, tau_path, where="post", alpha=0.85, label=r"Tauchen ($N=7$)")
    ax2c.set_xlabel("Period $t$")
    ax2c.set_ylabel("Shock state $z_t$")
    ax2c.set_title(r"Tauchen path")
    ax2c.legend(fontsize=9)

    # Bottom-right: Rouwenhorst path vs continuous AR(1)
    ax2d.plot(t_axis, true_path, color="black", linewidth=1.8, alpha=0.75, label="Continuous AR(1)")
    ax2d.step(t_axis, rou_path, where="post", color="tab:orange", alpha=0.85,
              label=r"Rouwenhorst ($N=7$)")
    ax2d.set_xlabel("Period $t$")
    ax2d.set_ylabel("Shock state $z_t$")
    ax2d.set_title(r"Rouwenhorst path")
    ax2d.legend(fontsize=9)

    fig2.tight_layout()
    save_figure(fig2, "figures/moment-accuracy.png", dpi=150)

    # Table: numerical detail behind the moment-accuracy figure
    table = comparison.copy()
    for col in ["Std", "Std error", "Persistence", "Persistence error"]:
        table[col] = table[col].map(lambda x: f"{x:.5f}")
    Path("tables").mkdir(parents=True, exist_ok=True)
    table.to_csv("tables/moment-comparison.csv", index=False)

    save_thumbnail("figures/stationary-mass.png", "figures/thumb.png")
    print("Generated figures and tables.")


if __name__ == "__main__":
    main()
