#!/usr/bin/env python3
"""IID income risk and buffer-stock saving by grid VFI.

The tutorial solves a partial-equilibrium household problem with uninsurable
IID income shocks. It is the next step after the deterministic household
problem: the same borrowing constraint now creates a precautionary asset buffer.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.grids import exponential_grid
from lib.output import ModelReport
from lib.plotting import setup_style


def crra_utility(c: np.ndarray, gamma: float) -> np.ndarray:
    """CRRA utility on positive consumption."""
    if gamma == 1.0:
        return np.log(c)
    return (c ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


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


def solve_iid_income_vfi(
    asset_grid: np.ndarray,
    income_grid: np.ndarray,
    income_probs: np.ndarray,
    beta: float,
    gross_return: float,
    gamma: float,
    tol: float,
    max_iter: int,
    label: str,
    verbose: bool = False,
) -> dict[str, np.ndarray | float | int | bool]:
    """Solve the discrete-grid IID income fluctuation problem by VFI."""
    n_asset = len(asset_grid)
    n_income = len(income_grid)
    cash_on_hand = gross_return * asset_grid[:, None] + income_grid[None, :]

    maintenance_consumption = (gross_return - 1.0) * asset_grid[:, None] + income_grid[None, :]
    value = crra_utility(maintenance_consumption, gamma) / (1.0 - beta)
    policy_idx = np.zeros((n_asset, n_income), dtype=int)
    row_idx = np.arange(n_asset)
    error = np.inf

    if verbose:
        print(f"\nStarting {label} VFI with {n_asset} asset grid points...")

    for iteration in range(1, max_iter + 1):
        value_next = np.empty_like(value)
        policy_idx_next = np.empty_like(policy_idx)
        expected_value = value @ income_probs

        for iy, _ in enumerate(income_grid):
            consumption = cash_on_hand[:, [iy]] - asset_grid[None, :]
            feasible = consumption > 1e-12
            values = (
                crra_utility(np.maximum(consumption, 1e-12), gamma)
                + beta * expected_value[None, :]
            )
            values[~feasible] = -np.inf
            best = np.argmax(values, axis=1)
            value_next[:, iy] = values[row_idx, best]
            policy_idx_next[:, iy] = best

        error = float(np.max(np.abs(value_next - value)))
        value = value_next
        policy_idx = policy_idx_next

        if verbose and (iteration % 50 == 0 or error < tol):
            print(f"  {label} iteration {iteration:4d}, error = {error:.2e}")
        if error < tol:
            break
    else:
        if verbose:
            print(f"  {label} did NOT converge after {max_iter} iterations")

    policy_assets = asset_grid[policy_idx]
    policy_consumption = cash_on_hand - policy_assets

    return {
        "value": value,
        "policy_idx": policy_idx,
        "policy_assets": policy_assets,
        "policy_consumption": policy_consumption,
        "iterations": iteration,
        "converged": error < tol,
        "error": error,
    }


def stationary_distribution(
    policy_idx: np.ndarray,
    income_probs: np.ndarray,
    tol: float = 1e-13,
    max_iter: int = 20_000,
) -> tuple[np.ndarray, int, float]:
    """Invariant distribution over asset and income states."""
    n_asset, n_income = policy_idx.shape
    dist = np.zeros((n_asset, n_income))
    dist[0, :] = income_probs
    error = np.inf

    for iteration in range(1, max_iter + 1):
        dist_next = np.zeros_like(dist)
        for iy in range(n_income):
            mass_by_next_asset = np.bincount(
                policy_idx[:, iy],
                weights=dist[:, iy],
                minlength=n_asset,
            )
            dist_next += mass_by_next_asset[:, None] * income_probs[None, :]
        error = float(np.max(np.abs(dist_next - dist)))
        dist = dist_next
        if error < tol:
            break

    return dist, iteration, error


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: list[float],
) -> np.ndarray:
    """Weighted quantiles for a discrete distribution."""
    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cumulative = np.cumsum(weights_sorted)
    cumulative = cumulative / cumulative[-1]
    return np.interp(quantiles, cumulative, values_sorted)


def simulate_panel(
    policy_idx: np.ndarray,
    asset_grid: np.ndarray,
    income_grid: np.ndarray,
    income_probs: np.ndarray,
    periods: int,
    n_agents: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate asset and income paths using the discrete policy."""
    rng = np.random.default_rng(seed)
    draws = rng.random((n_agents, periods))
    income_cdf = np.cumsum(income_probs)
    income_idx = np.searchsorted(income_cdf, draws, side="right")
    asset_idx = np.zeros((n_agents, periods), dtype=int)

    for t in range(periods - 1):
        asset_idx[:, t + 1] = policy_idx[asset_idx[:, t], income_idx[:, t]]

    assets = asset_grid[asset_idx]
    income = income_grid[income_idx]
    return assets, income


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
    n_asset = 550
    n_asset_refined = 1300
    tol = 1e-6
    max_iter = 2000

    # Simulation is used for paths. Distribution statistics come from the
    # invariant distribution induced by the discrete policy.
    n_agents = 200
    periods = 500

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
        exponential_grid(borrowing_limit, asset_max, n_asset_refined, density=3.0),
        dtype=float,
    )

    solution = solve_iid_income_vfi(
        asset_grid,
        income_grid,
        income_probs,
        beta,
        gross_return,
        gamma,
        tol,
        max_iter,
        label="main-grid",
        verbose=True,
    )
    refined_solution = solve_iid_income_vfi(
        asset_grid_refined,
        income_grid,
        income_probs,
        beta,
        gross_return,
        gamma,
        tol,
        max_iter,
        label="refined-grid",
        verbose=True,
    )

    value = np.asarray(solution["value"])
    policy_idx = np.asarray(solution["policy_idx"], dtype=int)
    policy_assets = np.asarray(solution["policy_assets"])
    policy_consumption = np.asarray(solution["policy_consumption"])
    savings_policy = policy_assets - asset_grid[:, None]

    refined_policy_consumption = np.asarray(refined_solution["policy_consumption"])
    refined_policy_assets = np.asarray(refined_solution["policy_assets"])
    mid_income_idx = n_income // 2
    refined_c_mid = np.interp(
        asset_grid,
        asset_grid_refined,
        refined_policy_consumption[:, mid_income_idx],
    )
    refined_a_mid = np.interp(
        asset_grid,
        asset_grid_refined,
        refined_policy_assets[:, mid_income_idx],
    )
    visible = asset_grid <= 5.0
    max_consumption_gap = float(
        np.max(np.abs(policy_consumption[visible, mid_income_idx] - refined_c_mid[visible]))
    )
    max_savings_gap = float(
        np.max(np.abs(policy_assets[visible, mid_income_idx] - refined_a_mid[visible]))
    )

    dist, dist_iterations, dist_error = stationary_distribution(
        policy_idx, income_probs
    )
    asset_dist = dist.sum(axis=1)
    mean_assets = float(asset_grid @ asset_dist)
    frac_constrained = float(asset_dist[0] * 100.0)
    pct_10, pct_50, pct_90, pct_99 = weighted_quantile(
        asset_grid / mean_income,
        asset_dist,
        [0.10, 0.50, 0.90, 0.99],
    )

    assets_sim, _ = simulate_panel(
        policy_idx,
        asset_grid,
        income_grid,
        income_probs,
        periods=periods,
        n_agents=n_agents,
        seed=2024,
    )
    final_sim_mean = float(np.mean(assets_sim[:, -1]))

    print("\nStationary distribution from discrete policy:")
    print(f"  Iterations: {dist_iterations}, error = {dist_error:.2e}")
    print(f"  Mean assets / mean income: {mean_assets / mean_income:.3f}")
    print(f"  Fraction at borrowing constraint: {frac_constrained:.1f}%")
    print(f"  Simulated final mean assets: {final_sim_mean:.3f}")

    setup_style()

    report = ModelReport(
        "IID Income Risk and Buffer-Stock Saving",
        "A partial-equilibrium household savings problem with uninsurable IID income shocks.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "The deterministic savings problem has no reason to hold a buffer stock when "
        "$\\beta R<1$: assets are eventually spent down to the borrowing limit. This "
        "tutorial changes only one economic object. Labor income is now random, "
        "uninsurable, and independent over time.\n\n"
        "That small change is enough to make assets useful. A household with low wealth "
        "is exposed to bad income draws because it cannot borrow below "
        "$\\underline a=0$. Saving is therefore not about financing retirement or "
        "aggregate capital accumulation here; it is self-insurance against the next "
        "income draw. The IID assumption keeps the expectation simple, so the tutorial "
        "isolates the buffer-stock motive before the later "
        "[endogenous-grid method](../endogenous-grid-points/) changes the solver and "
        "the persistent-income tutorial in "
        "[Dynamic Programming](../../dynamic-programming/consumption-savings/) changes "
        "the shock process."
    )

    report.add_equations(
        r"""
At the beginning of a period the household has assets $a \in A$ and receives
income $y_j$ from a finite IID distribution with probabilities $\pi_j$. It
chooses next-period assets $a' \in A$:

$$
V(a,y_j) =
\max_{a' \in A}
\{u(Ra+y_j-a') + \beta \sum_{\ell=1}^{n_y} \pi_{\ell} V(a',y_{\ell})\}.
$$

The budget identity and borrowing constraint are

$$
c = Ra + y_j - a',
\qquad
c>0,
\qquad
a' \geq \underline a.
$$

Preferences are CRRA,

$$
u(c)=
\begin{cases}
\dfrac{c^{1-\gamma}-1}{1-\gamma}, & \gamma \neq 1,\\[4pt]
\log c, & \gamma = 1.
\end{cases}
$$

Current income affects cash on hand. Because income is IID, it does not affect
beliefs about next period's income: the same probability vector $\pi$ is used
from every current income state.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Role |\n"
        f"|---|---:|---|\n"
        f"| $\\gamma$ | {gamma:.1f} | CRRA risk aversion |\n"
        f"| $\\beta$ | {beta:.2f} | Discount factor |\n"
        f"| $r$ | {r:.2f} | Net risk-free return |\n"
        f"| $\\beta R$ | {beta_r:.4f} | Patience-return product |\n"
        f"| $\\mu_y$ | {mean_income:.1f} | Mean income |\n"
        f"| $\\sigma_y$ | {sd_income:.1f} | Income standard deviation |\n"
        f"| $n_y$ | {n_income} | IID income states |\n"
        f"| $\\underline a$ | {borrowing_limit:.1f} | Borrowing limit |\n"
        f"| $\\bar a$ | {asset_max:.1f} | Upper asset-grid bound |\n"
        f"| Asset grid | {n_asset} points | Exponential spacing near the constraint |\n"
        f"| Refined grid | {n_asset_refined} points | Policy-function accuracy check |"
    )

    report.add_solution_method(
        "The code uses direct grid VFI. For each state $(a,y_j)$ it evaluates the "
        "lifetime value of every feasible next-asset choice. The only shortcut is an "
        "economic one: IID income means the continuation value depends on $a'$ but not "
        "on current income once the expectation over $y'$ has been taken.\n\n"
        "```text\n"
        "Input: asset grid A, income states y_j with probabilities pi_j, primitives beta, R, gamma\n"
        "Initialize V_0(a,y_j), for example from consuming interest income plus current y_j\n"
        "For n = 0, 1, 2, ...:\n"
        "    Compute EV_n(a') = sum_j pi_j V_n(a', y_j) for each candidate a'\n"
        "    For each current asset a in A and current income y_j:\n"
        "        For each candidate next asset a' in A:\n"
        "            Set c = R a + y_j - a'\n"
        "            If c <= 0, mark the choice infeasible\n"
        "            Otherwise compute u(c) + beta EV_n(a')\n"
        "        Store the maximizing next asset g(a,y_j) and value V_{n+1}(a,y_j)\n"
        "    Stop when max_{a,j} |V_{n+1}(a,y_j) - V_n(a,y_j)| < epsilon\n"
        "Output: value function V, savings policy g, consumption policy c(a,y_j)\n"
        "```\n\n"
        "After solving the policy, the stationary distribution is computed from the "
        "finite-state transition matrix implied by $g(a,y_j)$ and the IID income "
        "probabilities. The simulated paths are only used to visualize household-level "
        "asset histories.\n\n"
        f"The main grid converged in **{int(solution['iterations'])} iterations** "
        f"with sup-norm error {float(solution['error']):.2e}. A refined "
        f"{n_asset_refined}-point grid gives a median-income consumption policy within "
        f"{max_consumption_gap:.3e} over $a\\leq 5$; the corresponding next-asset gap is "
        f"{max_savings_gap:.3e}."
    )

    report.add_results(
        "The value functions are ordered by current income because high income raises "
        "current resources. The more interesting feature is that the gaps shrink with "
        "wealth: once assets are high, the current income draw is a smaller part of "
        "lifetime resources."
    )

    fig1, ax1 = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_income))
    for iy in range(n_income):
        ax1.plot(
            asset_grid,
            value[:, iy],
            color=colors[iy],
            linewidth=2,
            label=f"$y = {income_grid[iy]:.2f}$",
        )
    ax1.set_xlabel("Assets $a$")
    ax1.set_ylabel("$V(a,y)$")
    ax1.set_title("Value Functions")
    ax1.set_xlim(0, 5)
    ax1.legend()
    report.add_figure(
        "figures/value-functions.png",
        "Value functions by IID income state",
        fig1,
        description="",
    )

    report.add_results(
        "The consumption policy shows the buffer-stock mechanism directly. Low-wealth "
        "households consume a large share of cash on hand but do not behave as in the "
        "deterministic benchmark: even around the borrowing limit, a middle-income "
        "household saves a little because tomorrow's income may be bad. The dashed line "
        "is the refined-grid reference for the middle income state and nearly overlays "
        "the main-grid policy on the plotted range."
    )

    fig2, ax2 = plt.subplots()
    for iy in [0, mid_income_idx, n_income - 1]:
        ax2.plot(
            asset_grid,
            policy_consumption[:, iy],
            linewidth=2,
            label=f"$y = {income_grid[iy]:.2f}$",
        )
    ax2.plot(
        asset_grid,
        refined_c_mid,
        "k--",
        linewidth=1.3,
        label="refined grid, middle $y$",
    )
    ax2.set_xlabel("Assets $a$")
    ax2.set_ylabel("Consumption $c^{*}(a,y)$")
    ax2.set_title("Consumption Policy")
    ax2.set_xlim(0, 5)
    ax2.legend()
    report.add_figure(
        "figures/consumption-policy.png",
        "Consumption policy with refined-grid reference",
        fig2,
        description="",
    )

    report.add_results(
        "Net saving makes the insurance role of assets clearer than consumption alone. "
        "After a low income draw, the household runs down wealth. After a high draw, it "
        "rebuilds the buffer. The zero line is not a common steady state for all income "
        "states; with IID shocks the policy is state contingent even though the shock has "
        "no persistence."
    )

    fig3, ax3 = plt.subplots()
    for iy in [0, mid_income_idx, n_income - 1]:
        ax3.plot(
            asset_grid,
            savings_policy[:, iy],
            linewidth=2,
            label=f"$y = {income_grid[iy]:.2f}$",
        )
    ax3.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    ax3.set_xlabel("Assets $a$")
    ax3.set_ylabel("Net saving $a' - a$")
    ax3.set_title("Savings Policy")
    ax3.set_xlim(0, 5)
    ax3.legend()
    report.add_figure(
        "figures/savings-policy.png",
        "Net saving by current income state",
        fig3,
        description="",
    )

    report.add_results(
        "The path simulation starts all households at the borrowing limit. Individual "
        "histories move with income draws, while the cross-sectional mean settles near "
        "the invariant mean computed from the policy. The point is not aggregate risk; "
        "it is the stationary cross section produced by idiosyncratic self-insurance."
    )

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))
    for i in range(min(20, n_agents)):
        ax4a.plot(range(periods), assets_sim[i, :], linewidth=0.6, alpha=0.55)
    ax4a.axhline(mean_assets, color="k", linewidth=1.5, label="invariant mean")
    ax4a.set_xlabel("Period")
    ax4a.set_ylabel("Assets $a_t$")
    ax4a.set_title("Simulated Asset Histories")
    ax4a.set_xlim(0, periods)
    ax4a.legend()

    ax4b.plot(range(periods), np.mean(assets_sim, axis=0), linewidth=2)
    ax4b.axhline(mean_assets, color="k", linewidth=1.5, linestyle="--")
    ax4b.set_xlabel("Period")
    ax4b.set_ylabel("Mean assets")
    ax4b.set_title("Cross-Sectional Mean")
    fig4.tight_layout()
    report.add_figure(
        "figures/simulated-paths.png",
        "Simulated asset paths and convergence toward invariant mean",
        fig4,
        description="",
    )

    a_select = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    a_indices = [int(np.argmin(np.abs(asset_grid - a))) for a in a_select]
    table_data = {
        "Assets a": [f"{asset_grid[i]:.3f}" for i in a_indices],
    }
    for iy, label in [(0, "low y"), (mid_income_idx, "middle y"), (n_income - 1, "high y")]:
        table_data[f"c^{{*}}(a, {label})"] = [
            f"{policy_consumption[i, iy]:.4f}" for i in a_indices
        ]
    for iy, label in [(0, "low y"), (mid_income_idx, "middle y"), (n_income - 1, "high y")]:
        table_data[f"g(a, {label})"] = [
            f"{policy_assets[i, iy]:.4f}" for i in a_indices
        ]

    policy_table = pd.DataFrame(table_data)
    report.add_table(
        "tables/policy-values.csv",
        "Selected Policy Values",
        policy_table,
        description="The selected grid points emphasize the economically active region near "
        "the borrowing constraint. At $a=0$, the low-income household is constrained, "
        "but the middle- and high-income households still choose positive next-period "
        "assets because the next draw may be worse.",
    )

    stats_table = pd.DataFrame(
        {
            "Statistic": [
                "Mean assets / mean income",
                "Fraction at borrowing constraint",
                "10th percentile",
                "50th percentile",
                "90th percentile",
                "99th percentile",
                "Distribution iteration error",
            ],
            "Value": [
                f"{mean_assets / mean_income:.3f}",
                f"{frac_constrained:.1f}%",
                f"{pct_10:.3f}",
                f"{pct_50:.3f}",
                f"{pct_90:.3f}",
                f"{pct_99:.3f}",
                f"{dist_error:.1e}",
            ],
        }
    )
    report.add_table(
        "tables/simulation-stats.csv",
        "Invariant Asset Distribution",
        stats_table,
        description="The distribution is right-skewed, but it is not the persistent-income "
        "wealth distribution from an Aiyagari model. With IID risk the buffer is modest: "
        "many households remain close to the constraint, and high assets are rare. The "
        "statistics below use the exact invariant distribution of the discrete policy, "
        "not terminal-period Monte Carlo noise.",
    )

    report.add_takeaway(
        "IID income risk is the cleanest way to see the buffer-stock motive. The "
        "deterministic model says an impatient household should move back to the asset "
        "floor. Adding uninsurable income shocks overturns that conclusion: assets now "
        "pay an insurance return by protecting consumption after bad draws.\n\n"
        "The IID assumption also matters. Current income changes cash on hand and hence "
        "today's policy, but it does not change tomorrow's income distribution. Persistent "
        "income risk makes the state richer and the distribution more dispersed; the "
        "economic object here is the simpler benchmark that separates risk from "
        "persistence."
    )

    report.add_references([
        "Deaton, A. (1991). Saving and Liquidity Constraints. *Econometrica*, 59(5), 1221-1248.",
        "Carroll, C. (1997). Buffer-Stock Saving and the Life Cycle/Permanent Income Hypothesis. "
        "*Quarterly Journal of Economics*, 112(1), 1-55.",
        "Aiyagari, S. R. (1994). Uninsured Idiosyncratic Risk and Aggregate Saving. "
        "*Quarterly Journal of Economics*, 109(3), 659-684.",
        "Kaplan, G. (2017). *Heterogeneous Agent Models: Codes*. Lecture notes.",
    ])

    report.write("README.md")
    print(
        f"\nGenerated: README.md + {len(report._figures)} figures "
        f"+ {len(report._tables)} tables"
    )


if __name__ == "__main__":
    main()
