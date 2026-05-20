#!/usr/bin/env python3
"""Income fluctuation problem with persistent idiosyncratic income risk.

The tutorial solves a partial-equilibrium household savings problem by value
function iteration. It is the individual problem behind Aiyagari-style
incomplete-market models, before the interest rate is made endogenous.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.discretize import rouwenhorst
from lib.grids import exponential_grid
from lib.plotting import setup_style, save_figure, save_thumbnail


def crra_utility(c: np.ndarray, sigma: float) -> np.ndarray:
    """CRRA utility evaluated safely on positive consumption."""
    c_safe = np.maximum(c, 1e-15)
    if sigma == 1.0:
        return np.log(c_safe)
    return c_safe ** (1.0 - sigma) / (1.0 - sigma)


def solve_income_fluctuation(
    a_grid: np.ndarray,
    a_choice_grid: np.ndarray,
    z_grid: np.ndarray,
    transition: np.ndarray,
    beta: float,
    r: float,
    sigma: float,
    tol: float,
    max_iter: int,
    label: str,
    verbose: bool = False,
) -> dict[str, np.ndarray | float | int | bool]:
    """Solve the discrete-grid income fluctuation problem by VFI."""
    n_asset = len(a_grid)
    n_choice = len(a_choice_grid)
    n_income = len(z_grid)
    gross_return = 1.0 + r
    cash_on_hand = gross_return * a_grid[:, None] + z_grid[None, :]

    value = crra_utility(cash_on_hand, sigma) / (1.0 - beta)
    policy_a_idx = np.zeros((n_asset, n_income), dtype=int)
    row_idx = np.arange(n_asset)

    if verbose:
        print(
            f"\nStarting {label} VFI with {n_asset} asset states, "
            f"{n_choice} choices x {n_income} income states..."
        )

    for iteration in range(1, max_iter + 1):
        value_new = np.empty_like(value)
        policy_a_idx_new = np.empty_like(policy_a_idx)

        for iz in range(n_income):
            continuation_state = value @ transition[iz, :]
            continuation_choice = np.interp(a_choice_grid, a_grid, continuation_state)
            consumption = cash_on_hand[:, [iz]] - a_choice_grid[None, :]
            feasible = consumption > 1e-10

            values = crra_utility(consumption, sigma) + beta * continuation_choice[None, :]
            values[~feasible] = -np.inf

            best = np.argmax(values, axis=1)
            value_new[:, iz] = values[row_idx, best]
            policy_a_idx_new[:, iz] = best

        error = float(np.max(np.abs(value_new - value)))
        value = value_new
        policy_a_idx = policy_a_idx_new

        if verbose and iteration % 50 == 0:
            print(f"  {label} iteration {iteration:4d}, error = {error:.2e}")
        if error < tol:
            if verbose:
                print(f"  {label} converged in {iteration} iterations (error = {error:.2e})")
            break
    else:
        if verbose:
            print(f"  {label} did NOT converge after {max_iter} iterations (error = {error:.2e})")

    policy_a = a_choice_grid[policy_a_idx]
    policy_c = cash_on_hand - policy_a

    return {
        "value": value,
        "policy_a": policy_a,
        "policy_c": policy_c,
        "policy_idx": policy_a_idx,
        "iterations": iteration,
        "converged": error < tol,
        "error": error,
    }


def simulate_assets(
    policy_a: np.ndarray,
    a_grid: np.ndarray,
    transition: np.ndarray,
    initial_income_idx: np.ndarray,
    periods: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate assets and income states under a solved asset policy."""
    rng = np.random.default_rng(seed)
    n_agents = len(initial_income_idx)
    assets = np.zeros((periods, n_agents))
    income_idx = np.zeros((periods, n_agents), dtype=int)
    income_idx[0, :] = initial_income_idx

    for t in range(periods - 1):
        for agent in range(n_agents):
            iz = income_idx[t, agent]
            assets[t + 1, agent] = np.interp(assets[t, agent], a_grid, policy_a[:, iz])
            draw = rng.random()
            income_idx[t + 1, agent] = np.searchsorted(np.cumsum(transition[iz, :]), draw)

    return assets, income_idx


def main() -> None:
    # Preferences, prices, and constraints
    beta = 0.95
    r = 0.03
    gross_return = 1.0 + r
    sigma_crra = 2.0
    borrowing_limit = 0.0

    # Persistent log income
    rho = 0.9
    sigma_eps = 0.1
    n_income = 5

    # Asset grids
    n_asset = 300
    n_choice = 900
    n_asset_refined = 600
    n_choice_refined = 1500
    a_min = borrowing_limit
    a_max = 20.0

    # VFI settings
    tol = 1e-6
    max_iter = 2000

    z_grid_log_jax, trans_jax, ergodic_dist_jax = rouwenhorst(
        n=n_income, mu=0.0, sigma=sigma_eps, rho=rho
    )
    z_grid_log = np.asarray(z_grid_log_jax, dtype=float).flatten()
    z_grid = np.exp(z_grid_log)
    transition = np.asarray(trans_jax, dtype=float)
    ergodic_dist = np.asarray(ergodic_dist_jax, dtype=float).flatten()
    ergodic_dist = ergodic_dist / ergodic_dist.sum()

    print(f"Income grid (levels): {z_grid}")
    print(f"Ergodic distribution: {ergodic_dist}")
    print(f"Transition matrix:\n{transition}")

    a_grid = np.asarray(exponential_grid(a_min, a_max, n_asset, density=3.0), dtype=float)
    a_choice_grid = np.asarray(
        exponential_grid(a_min, a_max, n_choice, density=3.0),
        dtype=float,
    )
    a_grid_refined = np.asarray(
        exponential_grid(a_min, a_max, n_asset_refined, density=3.0),
        dtype=float,
    )
    a_choice_grid_refined = np.asarray(
        exponential_grid(a_min, a_max, n_choice_refined, density=3.0),
        dtype=float,
    )

    solution = solve_income_fluctuation(
        a_grid,
        a_choice_grid,
        z_grid,
        transition,
        beta,
        r,
        sigma_crra,
        tol,
        max_iter,
        label="main-grid",
        verbose=True,
    )
    refined_solution = solve_income_fluctuation(
        a_grid_refined,
        a_choice_grid_refined,
        z_grid,
        transition,
        beta,
        r,
        sigma_crra,
        tol,
        max_iter,
        label="refined-grid",
        verbose=True,
    )

    value = np.asarray(solution["value"])
    policy_a = np.asarray(solution["policy_a"])
    policy_c = np.asarray(solution["policy_c"])
    savings_policy = policy_a - a_grid[:, None]

    refined_policy_c = np.asarray(refined_solution["policy_c"])
    median_z_idx = n_income // 2
    refined_c_mid_on_main = np.interp(
        a_grid, a_grid_refined, refined_policy_c[:, median_z_idx]
    )
    refined_gap_mid = policy_c[:, median_z_idx] - refined_c_mid_on_main
    max_refined_gap_mid = float(np.max(np.abs(refined_gap_mid)))

    # MPC is the consumption response to cash-on-hand w = R a + z, not to a.
    # Since dw/da = R, the marginal propensity to consume out of cash-on-hand
    # is dc/dw = (dc/da) / R.
    dc_da_mid = np.gradient(policy_c[:, median_z_idx], a_grid)
    mpc_mid = dc_da_mid / gross_return
    low_asset_mpc = float(np.mean(mpc_mid[5:20]))
    high_asset_mpc = float(np.mean(mpc_mid[-40:]))

    initial_paths_income = np.full(5, median_z_idx)
    path_assets, _ = simulate_assets(
        policy_a, a_grid, transition, initial_paths_income, periods=200, seed=42
    )

    rng = np.random.default_rng(123)
    n_panel_agents = 3000
    initial_panel_income = rng.choice(n_income, size=n_panel_agents, p=ergodic_dist)
    panel_assets, _ = simulate_assets(
        policy_a, a_grid, transition, initial_panel_income, periods=400, seed=1234
    )
    final_assets = panel_assets[-1, :]
    median_assets = float(np.median(final_assets))
    p90_assets = float(np.quantile(final_assets, 0.9))
    constraint_share = float(np.mean(final_assets <= a_grid[1]))

    setup_style()

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_income))

    fig1, ax1 = plt.subplots()
    for iz in range(n_income):
        ax1.plot(a_grid, value[:, iz], color=colors[iz], linewidth=2, label=f"$z={z_grid[iz]:.3f}$")
    ax1.set_xlabel("Assets $a$")
    ax1.set_ylabel("$V(a,z)$")
    ax1.set_title("Value by Income State")
    ax1.legend(fontsize=9)
    save_figure(fig1, "figures/value-functions.png", dpi=150)

    fig2, ax2 = plt.subplots()
    for iz in range(n_income):
        ax2.plot(a_grid, policy_c[:, iz], color=colors[iz], linewidth=2, label=f"$z={z_grid[iz]:.3f}$")
    ax2.plot(
        a_grid,
        refined_c_mid_on_main,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="refined benchmark, median $z$",
    )
    ax2.set_xlabel("Assets $a$")
    ax2.set_ylabel("Consumption $c^{*}(a,z)$")
    ax2.set_title("Consumption Policy")
    ax2.legend(fontsize=8)
    save_figure(fig2, "figures/consumption-policy.png", dpi=150)

    fig3, ax3 = plt.subplots()
    for iz in range(n_income):
        ax3.plot(a_grid, savings_policy[:, iz], color=colors[iz], linewidth=2, label=f"$z={z_grid[iz]:.3f}$")
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax3.set_xlabel("Assets $a$")
    ax3.set_ylabel("Net saving $g_a(a,z)-a$")
    ax3.set_title("Net Saving Policy")
    ax3.legend(fontsize=9)
    save_figure(fig3, "figures/savings-policy.png", dpi=150)

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))
    periods = np.arange(path_assets.shape[0])
    for agent in range(path_assets.shape[1]):
        ax4a.plot(periods, path_assets[:, agent], linewidth=1.1, alpha=0.85, label=f"Agent {agent + 1}")
    ax4a.set_xlabel("Period")
    ax4a.set_ylabel("Assets $a_t$")
    ax4a.set_title("Five Sample Histories")
    ax4a.legend(fontsize=8)

    ax4b.hist(final_assets, bins=35, color="#4C78A8", edgecolor="white", alpha=0.9)
    ax4b.axvline(median_assets, color="black", linestyle="--", linewidth=1.2, label="Median")
    ax4b.axvline(p90_assets, color="black", linestyle=":", linewidth=1.2, label="90th pct.")
    ax4b.set_xlabel("Assets after 400 periods")
    ax4b.set_ylabel("Agents")
    ax4b.set_title("Simulated Cross-Section")
    ax4b.legend(fontsize=8)
    fig4.tight_layout()
    save_figure(fig4, "figures/simulated-paths.png", dpi=150)

    # Thumbnail
    save_thumbnail("figures/value-functions.png", "figures/thumb.png")

    # =========================================================================
    # Table
    # =========================================================================
    scalars_df = pd.DataFrame(
        {
            "Quantity": [
                "Main-grid VFI iterations",
                "Main-grid sup-norm residual",
                "Refined-grid max gap (median z)",
                "MPC near zero assets (median z)",
                "MPC near top assets (median z)",
                "Simulated median wealth",
                "Simulated P90 wealth",
                "Share near constraint",
            ],
            "Value": [
                f"{solution['iterations']}",
                f"{solution['error']:.2e}",
                f"{max_refined_gap_mid:.2e}",
                f"{low_asset_mpc:.4f}",
                f"{high_asset_mpc:.4f}",
                f"{median_assets:.4f}",
                f"{p90_assets:.4f}",
                f"{constraint_share:.4f}",
            ],
        }
    )
    Path("tables").mkdir(parents=True, exist_ok=True)
    scalars_df.to_csv("tables/key-scalars.csv", index=False)

    print(f"\nGenerated: figures + tables")


if __name__ == "__main__":
    main()
