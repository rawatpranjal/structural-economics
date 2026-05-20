#!/usr/bin/env python3
"""Endogenous grid points for an income-risk saving problem.

The tutorial uses a partial-equilibrium IID income-risk household problem and
solves the saving policy by Euler-equation inversion.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.grids import exponential_grid
from lib.plotting import setup_style, save_figure, save_thumbnail


def marginal_utility(consumption: np.ndarray, gamma: float) -> np.ndarray:
    """CRRA marginal utility."""
    return consumption ** (-gamma)


def inverse_marginal_utility(mu: np.ndarray, gamma: float) -> np.ndarray:
    """Inverse CRRA marginal utility."""
    return mu ** (-1.0 / gamma)


def discrete_normal_np(
    n: int,
    mu: float,
    sigma: float,
    width: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Equally spaced approximation to a normal distribution."""
    grid = np.linspace(mu - width * sigma, mu + width * sigma, n)
    if n == 2:
        probs = 0.5 * np.ones(n)
    else:
        probs = np.zeros(n)
        half_steps = 0.5 * np.diff(grid)
        probs[0] = norm.cdf(grid[0] + half_steps[0], mu, sigma)
        for i in range(1, n - 1):
            right = grid[i] + half_steps[i]
            left = grid[i] - half_steps[i - 1]
            probs[i] = norm.cdf(right, mu, sigma) - norm.cdf(left, mu, sigma)
        probs[-1] = 1.0 - np.sum(probs[:-1])

    mean = float(grid @ probs)
    sd = float(np.sqrt((grid ** 2) @ probs - mean ** 2))
    return sd - sigma, grid, probs


def build_income_grid(
    n_income: int,
    mean_income: float,
    sd_income: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Choose normal-grid width so the discrete standard deviation matches."""

    def objective(width: np.ndarray) -> float:
        error, _, _ = discrete_normal_np(
            n_income, mean_income, sd_income, float(width[0])
        )
        return error

    width = float(fsolve(objective, np.array([2.0]))[0])
    _, income_grid, income_probs = discrete_normal_np(
        n_income, mean_income, sd_income, width
    )
    return income_grid, income_probs, width


def solve_egp(
    asset_grid: np.ndarray,
    income_grid: np.ndarray,
    income_probs: np.ndarray,
    beta: float,
    gross_return: float,
    gamma: float,
    borrowing_limit: float,
    tol: float,
    max_iter: int,
    label: str,
    verbose: bool = False,
) -> dict[str, np.ndarray | float | int | bool]:
    """Solve the IID income fluctuation problem by endogenous grid points."""
    n_asset = len(asset_grid)
    n_income = len(income_grid)
    consumption = (gross_return - 1.0) * asset_grid[:, None] + income_grid[None, :]
    policy_assets = np.zeros((n_asset, n_income))
    endogenous_assets = np.zeros((n_asset, n_income))
    error = np.inf

    if verbose:
        print(f"\nStarting {label} EGP with {n_asset} asset grid points...")

    for iteration in range(1, max_iter + 1):
        consumption_old = consumption.copy()

        expected_mu = marginal_utility(consumption_old, gamma) @ income_probs
        euler_mu = beta * gross_return * expected_mu
        consumption_at_next_asset = inverse_marginal_utility(euler_mu, gamma)

        for iy, income in enumerate(income_grid):
            implied_assets = (
                consumption_at_next_asset + asset_grid - income
            ) / gross_return
            endogenous_assets[:, iy] = implied_assets

            policy = np.interp(asset_grid, implied_assets, asset_grid)
            policy[asset_grid <= implied_assets[0]] = borrowing_limit
            policy_assets[:, iy] = np.clip(policy, borrowing_limit, asset_grid[-1])
            consumption[:, iy] = (
                gross_return * asset_grid + income - policy_assets[:, iy]
            )

        error = float(np.max(np.abs(consumption - consumption_old)))
        if verbose and (iteration % 50 == 0 or error < tol):
            print(f"  {label} iteration {iteration:4d}, error = {error:.2e}")
        if error < tol:
            break
    else:
        if verbose:
            print(f"  {label} did NOT converge after {max_iter} iterations")

    return {
        "consumption": consumption,
        "policy_assets": policy_assets,
        "endogenous_assets": endogenous_assets,
        "iterations": iteration,
        "converged": error < tol,
        "error": error,
    }


def simulate_terminal_cross_section(
    asset_grid: np.ndarray,
    policy_assets: np.ndarray,
    income_probs: np.ndarray,
    n_agents: int,
    periods: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the terminal asset and income states under the policy."""
    rng = np.random.default_rng(seed)
    income_cdf = np.cumsum(income_probs)
    assets = np.zeros(n_agents)
    income_idx = np.zeros(n_agents, dtype=int)

    for t in range(periods):
        income_idx = np.searchsorted(income_cdf, rng.random(n_agents), side="right")
        if t == periods - 1:
            break

        assets_next = np.empty_like(assets)
        for iy in range(len(income_probs)):
            mask = income_idx == iy
            assets_next[mask] = np.interp(
                assets[mask], asset_grid, policy_assets[:, iy]
            )
        assets = assets_next

    return assets, income_idx


def gini(x: np.ndarray) -> float:
    """Gini coefficient for a nonnegative vector."""
    x_sorted = np.sort(np.asarray(x, dtype=float))
    total = np.sum(x_sorted)
    if total <= 0:
        return 0.0
    n = len(x_sorted)
    weights = np.arange(1, n + 1)
    return float((2.0 * weights @ x_sorted) / (n * total) - (n + 1.0) / n)


def policy_gap_against_reference(
    asset_grid: np.ndarray,
    policy: np.ndarray,
    reference_grid: np.ndarray,
    reference_policy: np.ndarray,
    upper_asset: float,
) -> float:
    """Maximum absolute policy gap on the economically active asset range."""
    visible = asset_grid <= upper_asset
    gaps = []
    for iy in range(policy.shape[1]):
        reference_on_main_grid = np.interp(
            asset_grid[visible], reference_grid, reference_policy[:, iy]
        )
        gaps.append(np.max(np.abs(policy[visible, iy] - reference_on_main_grid)))
    return float(np.max(gaps))


def main() -> None:
    # Preferences, returns, and income risk
    gamma = 2.0
    beta = 0.95
    r = 0.03
    gross_return = 1.0 + r
    beta_r = beta * gross_return

    mean_income = 1.0
    sd_income = 0.2
    n_income = 5

    # Assets and computation
    borrowing_limit = 0.0
    asset_max = 20.0
    n_asset = 120
    n_asset_refined = 900
    tol = 1e-6
    max_iter = 1000
    accuracy_asset_max = 5.0

    # Simulation for the stationary cross section
    n_agents = 50_000
    periods = 550
    seed = 2020

    income_grid, income_probs, width = build_income_grid(
        n_income, mean_income, sd_income
    )
    print(f"Income grid: {income_grid}")
    print(f"Income probabilities: {income_probs}")
    print(f"Normal-grid width parameter: {width:.4f}")

    asset_grid = np.asarray(
        exponential_grid(borrowing_limit, asset_max, n_asset, density=3.0),
        dtype=float,
    )
    asset_grid_refined = np.asarray(
        exponential_grid(
            borrowing_limit, asset_max, n_asset_refined, density=3.0
        ),
        dtype=float,
    )

    solution = solve_egp(
        asset_grid,
        income_grid,
        income_probs,
        beta,
        gross_return,
        gamma,
        borrowing_limit,
        tol,
        max_iter,
        label="main-grid",
        verbose=True,
    )
    refined_solution = solve_egp(
        asset_grid_refined,
        income_grid,
        income_probs,
        beta,
        gross_return,
        gamma,
        borrowing_limit,
        tol,
        max_iter,
        label="refined-grid",
        verbose=True,
    )

    consumption = np.asarray(solution["consumption"])
    policy_assets = np.asarray(solution["policy_assets"])
    endogenous_assets = np.asarray(solution["endogenous_assets"])
    refined_consumption = np.asarray(refined_solution["consumption"])
    refined_policy_assets = np.asarray(refined_solution["policy_assets"])

    consumption_gap = policy_gap_against_reference(
        asset_grid,
        consumption,
        asset_grid_refined,
        refined_consumption,
        upper_asset=accuracy_asset_max,
    )
    savings_gap = policy_gap_against_reference(
        asset_grid,
        policy_assets,
        asset_grid_refined,
        refined_policy_assets,
        upper_asset=accuracy_asset_max,
    )

    final_assets, final_income_idx = simulate_terminal_cross_section(
        asset_grid,
        policy_assets,
        income_probs,
        n_agents=n_agents,
        periods=periods,
        seed=seed,
    )

    con_interp = [
        interp1d(
            asset_grid,
            consumption[:, iy],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        for iy in range(n_income)
    ]

    final_consumption = np.zeros(n_agents)
    mpc_small = np.zeros(n_agents)
    mpc_large = np.zeros(n_agents)
    transfer_small = 1.0e-6
    transfer_large = 0.10
    for iy in range(n_income):
        mask = final_income_idx == iy
        assets_iy = final_assets[mask]
        final_consumption[mask] = con_interp[iy](assets_iy)
        mpc_small[mask] = (
            con_interp[iy](assets_iy + transfer_small) - con_interp[iy](assets_iy)
        ) / transfer_small
        mpc_large[mask] = (
            con_interp[iy](assets_iy + transfer_large) - con_interp[iy](assets_iy)
        ) / transfer_large

    mean_assets = float(np.mean(final_assets))
    mean_consumption = float(np.mean(final_consumption))
    gini_wealth = gini(final_assets)
    mean_mpc_small = float(np.mean(mpc_small))
    mean_mpc_large = float(np.mean(mpc_large))
    frac_constrained = float(np.mean(final_assets <= borrowing_limit + 1e-8) * 100.0)
    # Carroll (1997) perfect-foresight MPC limit: kappa* = 1 - (beta*R)^(1/gamma) / R
    # where G_c = (beta*R)^(1/gamma) is the consumption growth factor.
    # The growth-impatience condition G_c < R (i.e. beta*R < 1) ensures a finite
    # buffer-stock target.
    mpclim = 1.0 - (beta_r ** (1.0 / gamma)) / gross_return

    print("\nStationary cross section from simulation:")
    print(f"  Mean assets:              {mean_assets:.3f}")
    print(f"  Mean consumption:         {mean_consumption:.3f}")
    print(f"  Gini (wealth):            {gini_wealth:.3f}")
    print(f"  Average MPC, large shock: {mean_mpc_large:.3f}")
    print(f"  Fraction constrained:     {frac_constrained:.1f}%")
    print(f"  Perfect-foresight MPC limit (Carroll): {mpclim:.4f}")
    print(f"  Consumption gap vs fine grid on a <= {accuracy_asset_max:g}: {consumption_gap:.2e}")
    print(f"  Savings gap vs fine grid on a <= {accuracy_asset_max:g}: {savings_gap:.2e}")

    setup_style()

    plot_max = 8.0
    low = 0
    high = n_income - 1

    # Consumption policy
    fig1, ax1 = plt.subplots()
    ax1.plot(asset_grid, consumption[:, low], linewidth=2, label="Lowest income $y_1$")
    ax1.plot(asset_grid, consumption[:, high], linewidth=2, label="Highest income $y_{n_y}$")
    for iy in range(1, n_income - 1):
        ax1.plot(asset_grid, consumption[:, iy], color="gray", linewidth=0.8, alpha=0.45)
    ax1.plot(
        asset_grid_refined,
        refined_consumption[:, low],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Fine grid ($N_a={n_asset_refined}$)",
    )
    ax1.plot(
        asset_grid_refined,
        refined_consumption[:, high],
        color="black",
        linestyle="--",
        linewidth=1.2,
    )
    ax1.set_xlabel("Assets $a$")
    ax1.set_ylabel("Consumption $c(a,y)$")
    ax1.set_title("Consumption Policy")
    ax1.set_xlim(0, plot_max)
    ax1.legend()
    save_figure(fig1, "figures/consumption-policy.png", dpi=150)

    # Savings policy
    fig2, ax2 = plt.subplots()
    ax2.plot(
        asset_grid,
        policy_assets[:, low] - asset_grid,
        linewidth=2,
        label="Lowest income",
    )
    ax2.plot(
        asset_grid,
        policy_assets[:, high] - asset_grid,
        linewidth=2,
        label="Highest income",
    )
    for iy in range(1, n_income - 1):
        ax2.plot(
            asset_grid,
            policy_assets[:, iy] - asset_grid,
            color="gray",
            linewidth=0.8,
            alpha=0.45,
        )
    ax2.plot(
        asset_grid_refined,
        refined_policy_assets[:, low] - asset_grid_refined,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Fine-grid reference",
    )
    ax2.plot(
        asset_grid_refined,
        refined_policy_assets[:, high] - asset_grid_refined,
        color="black",
        linestyle="--",
        linewidth=1.2,
    )
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_xlabel("Assets $a$")
    ax2.set_ylabel("Net saving $g(a,y)-a$")
    ax2.set_title("Net Saving Policy")
    ax2.set_xlim(0, plot_max)
    ax2.legend()
    save_figure(fig2, "figures/savings-policy.png", dpi=150)

    # Endogenous grid map for the low income state
    fig3, ax3 = plt.subplots()
    ax3.plot(
        asset_grid,
        endogenous_assets[:, low],
        linewidth=2,
        label=r"$a^{\mathrm{endo}}_{i,1}$",
    )
    ax3.plot(asset_grid, asset_grid, color="black", linestyle=":", linewidth=1.2,
             label=r"$a^{\mathrm{endo}}=a'$")
    ax3.axhline(borrowing_limit, color="black", linewidth=0.8)
    ax3.set_xlabel("Candidate next assets $a'$")
    ax3.set_ylabel(r"Implied current assets $a^{\mathrm{endo}}$")
    ax3.set_title("Endogenous Grid Map, Lowest Income State")
    ax3.set_xlim(0, plot_max)
    ax3.set_ylim(-0.5, plot_max)
    ax3.legend()
    save_figure(fig3, "figures/endogenous-grid.png", dpi=150)

    # Wealth distribution
    fig4, ax4 = plt.subplots()
    upper_hist = float(max(2.0, np.quantile(final_assets, 0.995)))
    ax4.hist(
        final_assets,
        bins=90,
        density=True,
        color="steelblue",
        edgecolor="black",
        linewidth=0.25,
        alpha=0.85,
    )
    ax4.axvline(mean_assets, color="darkred", linestyle="--", linewidth=1.4,
                label=f"Mean = {mean_assets:.2f}")
    ax4.set_xlabel("Assets $a$")
    ax4.set_ylabel("Density")
    ax4.set_title("Simulated Stationary Wealth Distribution")
    ax4.set_xlim(0, upper_hist)
    ax4.legend()
    save_figure(fig4, "figures/wealth-distribution.png", dpi=150)

    # MPC distribution
    fig5, ax5 = plt.subplots()
    ax5.hist(
        mpc_large,
        bins=np.linspace(0, 1.05, 70),
        density=True,
        color="steelblue",
        edgecolor="black",
        linewidth=0.25,
        alpha=0.85,
    )
    ax5.axvline(mpclim, color="darkred", linestyle=":", linewidth=1.5,
                label=f"Perfect-foresight limit = {mpclim:.4f}")
    ax5.axvline(mean_mpc_large, color="darkorange", linestyle="--", linewidth=1.5,
                label=f"Mean = {mean_mpc_large:.3f}")
    ax5.set_xlabel("MPC out of a 0.10 transfer")
    ax5.set_ylabel("Density")
    ax5.set_title("Marginal Propensity to Consume")
    ax5.set_xlim(0, 1.05)
    ax5.legend()
    save_figure(fig5, "figures/mpc-distribution.png", dpi=150)

    table_data = {
        "Statistic": [
            "Mean assets",
            "Mean consumption",
            "Wealth Gini",
            "Average MPC, 0.10 transfer",
            "Average local consumption slope dc/da (numerical derivative, 1e-6 step)",
            "Fraction at borrowing limit",
            "Consumption gap vs fine grid, a <= 5",
            "Savings gap vs fine grid, a <= 5",
            "Perfect-foresight MPC limit",
        ],
        "Value": [
            f"{mean_assets:.3f}",
            f"{mean_consumption:.3f}",
            f"{gini_wealth:.3f}",
            f"{mean_mpc_large:.3f}",
            f"{mean_mpc_small:.3f}",
            f"{frac_constrained:.1f}%",
            f"{consumption_gap:.2e}",
            f"{savings_gap:.2e}",
            f"{mpclim:.4f}",
        ],
    }
    df = pd.DataFrame(table_data)
    Path("tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("tables/summary-statistics.csv", index=False)

    save_thumbnail("figures/consumption-policy.png", "figures/thumb.png")
    print(
        f"\nDone: 5 figures, 1 table. Mean assets={mean_assets:.3f}, "
        f"MPC={mean_mpc_large:.3f}, MPC limit={mpclim:.4f}"
    )


if __name__ == "__main__":
    main()
