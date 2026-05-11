#!/usr/bin/env python3
"""Aiyagari (1994) stationary equilibrium.

Solves an incomplete-markets economy where households face uninsurable
idiosyncratic productivity risk, save through a single risk-free asset, and
trade with a representative firm operating Cobb-Douglas technology. The
interest rate is the price that clears aggregate household savings against
firm capital demand. A sweep over r traces the household capital-supply
schedule against analytic capital demand.

Reference: Aiyagari, S. R. (1994). "Uninsured Idiosyncratic Risk and
Aggregate Saving." Quarterly Journal of Economics, 109(3), 659-684.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.discretize import rouwenhorst
from lib.grids import exponential_grid
from lib.output import ModelReport
from lib.plotting import setup_style


def crra_utility(c: np.ndarray, sigma: float) -> np.ndarray:
    """CRRA utility evaluated safely on positive consumption."""
    c_safe = np.maximum(c, 1e-15)
    if sigma == 1.0:
        return np.log(c_safe)
    return c_safe ** (1.0 - sigma) / (1.0 - sigma)


def solve_household(
    a_grid: np.ndarray,
    z_grid: np.ndarray,
    transition: np.ndarray,
    beta: float,
    sigma: float,
    r: float,
    w: float,
    tol: float = 1e-7,
    max_iter: int = 4000,
    value_init: np.ndarray | None = None,
) -> dict[str, np.ndarray | int | float | bool]:
    """Solve the household consumption-savings problem at prices (r, w).

    Vectorized grid-search VFI: cash on hand and the (current_asset,
    next_asset) consumption tensor are precomputed once; each iteration is a
    single argmax over the next-asset axis. Infeasible (negative-consumption)
    cells are masked with -infty.
    """
    n_asset = a_grid.size
    n_income = z_grid.size
    cash_on_hand = (1.0 + r) * a_grid[:, None] + w * z_grid[None, :]  # (n_a, n_z)
    consumption = cash_on_hand[:, :, None] - a_grid[None, None, :]   # (n_a, n_z, n_a')
    feasible = consumption > 1e-12

    if value_init is not None:
        value = value_init.copy()
    else:
        value = crra_utility(np.maximum(cash_on_hand, 1e-12), sigma) / (1.0 - beta)

    flow_utility = crra_utility(np.maximum(consumption, 1e-15), sigma)
    for iteration in range(1, max_iter + 1):
        # E[V(a', z') | z_j] over the income chain, indexed [m, j].
        expected_value = value @ transition.T
        # candidate[i, j, m] = u(c_{ijm}) + beta * E[V(a_m, z')|z_j]
        candidate = flow_utility + beta * expected_value.T[None, :, :]
        candidate = np.where(feasible, candidate, -np.inf)
        policy_idx = np.argmax(candidate, axis=2)
        value_new = np.take_along_axis(candidate, policy_idx[:, :, None], axis=2).squeeze(2)
        error = float(np.max(np.abs(value_new - value)))
        value = value_new
        if error < tol:
            break

    asset_policy = a_grid[policy_idx]
    consumption_policy = cash_on_hand - asset_policy
    return {
        "value": value,
        "policy_idx": policy_idx,
        "asset_policy": asset_policy,
        "consumption_policy": consumption_policy,
        "iterations": iteration,
        "error": error,
        "converged": error < tol,
    }


def stationary_distribution(
    policy_idx: np.ndarray,
    transition: np.ndarray,
    tol: float = 1e-11,
    max_iter: int = 10_000,
) -> tuple[np.ndarray, int, float]:
    """Forward-iterate the joint distribution over (a, z) under (g_a, P)."""
    n_asset, n_income = policy_idx.shape
    dist = np.full((n_asset, n_income), 1.0 / (n_asset * n_income))

    for iteration in range(1, max_iter + 1):
        dist_new = np.zeros_like(dist)
        for j in range(n_income):
            mass_at_j = dist[:, j]
            np.add.at(dist_new, (policy_idx[:, j], slice(None)),
                      mass_at_j[:, None] * transition[j, None, :])
        error = float(np.max(np.abs(dist_new - dist)))
        dist = dist_new
        if error < tol:
            break
    return dist, iteration, error


def aggregate_capital(dist: np.ndarray, a_grid: np.ndarray) -> float:
    return float(np.sum(dist * a_grid[:, None]))


def lorenz_and_gini(a_grid: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Population-share, wealth-share, and Gini from the (a, z) distribution."""
    n_asset, n_income = dist.shape
    asset_marginal = dist.sum(axis=1)
    sort_idx = np.argsort(a_grid)
    a_sorted = a_grid[sort_idx]
    mass_sorted = asset_marginal[sort_idx]
    cum_pop = np.cumsum(mass_sorted)
    cum_wealth = np.cumsum(mass_sorted * a_sorted)
    total_wealth = cum_wealth[-1]
    if total_wealth <= 0:
        return cum_pop, np.zeros_like(cum_wealth), 0.0
    wealth_share = cum_wealth / total_wealth
    # Gini via trapezoidal integration of the Lorenz curve.
    pop = np.concatenate([[0.0], cum_pop])
    share = np.concatenate([[0.0], wealth_share])
    gini = 1.0 - 2.0 * np.trapezoid(share, pop)
    return cum_pop, wealth_share, float(gini)


def main() -> None:
    # =========================================================================
    # Calibration
    # =========================================================================
    beta = 0.96
    sigma_crra = 2.0
    alpha = 0.36
    delta = 0.08
    rho = 0.9
    sigma_eps = 0.2
    n_income = 7

    n_asset = 200
    a_min = 0.0
    a_max = 50.0

    # Outer search: bisection on r.
    impatience_rate = 1.0 / beta - 1.0
    r_low_init = 0.005
    r_high_init = impatience_rate - 1e-3
    r_low, r_high = r_low_init, r_high_init
    tol_r = 5e-4
    tol_r_interval = 1e-6
    max_iter_ge = 60

    tol_vfi = 1e-7

    # =========================================================================
    # Income process and grids
    # =========================================================================
    z_grid_log_jax, transition_jax, ergodic_jax = rouwenhorst(
        n=n_income, mu=0.0, sigma=sigma_eps, rho=rho,
    )
    z_grid_log = np.asarray(z_grid_log_jax).ravel()
    z_grid = np.exp(z_grid_log)
    transition = np.asarray(transition_jax)
    ergodic = np.asarray(ergodic_jax).ravel()
    z_grid = z_grid / float(np.dot(ergodic, z_grid))   # normalize so E[z]=1
    L_supply = 1.0

    a_grid = np.asarray(exponential_grid(a_min, a_max, n_asset, density=3.0))

    print("Income grid (levels, normalized):", np.round(z_grid, 4))

    # =========================================================================
    # Firm side
    # =========================================================================
    def capital_demand(r_val: float) -> float:
        """K^d(r) = ((r+delta)/alpha)^{1/(alpha-1)} with L=1."""
        return L_supply * ((r_val + delta) / alpha) ** (1.0 / (alpha - 1.0))

    def wage(K: float) -> float:
        return (1.0 - alpha) * (K / L_supply) ** alpha

    # =========================================================================
    # Outer bisection on r
    # =========================================================================
    print(f"\nBisecting r in [{r_low_init:.4f}, {r_high_init:.4f}]")
    r_history: list[float] = []
    Ks_history: list[float] = []
    Kd_history: list[float] = []
    value_warm: np.ndarray | None = None

    for ge_iter in range(1, max_iter_ge + 1):
        r_trial = 0.5 * (r_low + r_high)
        K_d = capital_demand(r_trial)
        w_trial = wage(K_d)

        sol = solve_household(
            a_grid, z_grid, transition, beta, sigma_crra, r_trial, w_trial,
            tol=tol_vfi, value_init=value_warm,
        )
        value_warm = sol["value"]
        dist, _, _ = stationary_distribution(sol["policy_idx"], transition)
        K_s = aggregate_capital(dist, a_grid)

        r_history.append(r_trial)
        Ks_history.append(K_s)
        Kd_history.append(K_d)

        gap_rel = (K_s - K_d) / K_d
        print(f"  iter {ge_iter:2d}: r={r_trial:.6f}, "
              f"K^s={K_s:.4f}, K^d={K_d:.4f}, rel gap={gap_rel:+.3e}")

        if abs(gap_rel) < tol_r:
            ge_stop_reason = "the capital-market gap met the relative tolerance"
            break
        if gap_rel > 0:
            r_high = r_trial   # excess saving → r too high
        else:
            r_low = r_trial    # excess demand → r too low
        if r_high - r_low < tol_r_interval:
            ge_stop_reason = (
                "the bisection bracket narrowed below 1e-6, the discrete-grid "
                "supply schedule cannot move continuously through the demand curve"
            )
            break
    else:
        ge_stop_reason = "the iteration cap was reached"

    r_eq = r_trial
    K_eq = K_d
    w_eq = w_trial
    market_gap = K_s - K_d
    market_gap_rel = market_gap / K_d
    Y_eq = K_eq ** alpha * L_supply ** (1.0 - alpha)

    # =========================================================================
    # Distributional statistics at equilibrium
    # =========================================================================
    asset_marginal = dist.sum(axis=1)
    mean_wealth = K_s
    cum_pop, wealth_share, gini = lorenz_and_gini(a_grid, dist)
    sort_idx = np.argsort(a_grid)
    a_sorted = a_grid[sort_idx]
    mass_sorted = asset_marginal[sort_idx]
    cum_pop_full = np.cumsum(mass_sorted)
    median_wealth = float(a_sorted[np.searchsorted(cum_pop_full, 0.5)])
    p90_wealth = float(a_sorted[np.searchsorted(cum_pop_full, 0.9)])
    frac_constrained = float(dist[0, :].sum())

    # =========================================================================
    # Dense capital-supply curve for the equilibrium picture
    # =========================================================================
    n_supply_grid = 16
    r_supply_grid = np.linspace(r_low_init, r_high_init - 1e-4, n_supply_grid)
    Ks_curve = np.empty(n_supply_grid)
    value_warm_curve: np.ndarray | None = sol["value"]
    # Sweep upward in r so warm starts are well-conditioned.
    for idx, r_val in enumerate(r_supply_grid):
        K_d_local = capital_demand(r_val)
        w_local = wage(K_d_local)
        sol_local = solve_household(
            a_grid, z_grid, transition, beta, sigma_crra, r_val, w_local,
            tol=1e-6, value_init=value_warm_curve,
        )
        value_warm_curve = sol_local["value"]
        dist_local, _, _ = stationary_distribution(sol_local["policy_idx"], transition)
        Ks_curve[idx] = aggregate_capital(dist_local, a_grid)
    Kd_curve_dense = np.array([capital_demand(r) for r in r_supply_grid])

    # =========================================================================
    # Report
    # =========================================================================
    setup_style()
    report = ModelReport(
        "Aiyagari Saving and Capital-Market Clearing",
        include_reproduce=False,
        show_figure_captions=False,
    )

    K_over_Y = K_eq / Y_eq

    report.add_overview(
        "In Aiyagari (1994), households face persistent idiosyncratic income "
        "risk and cannot borrow. They save in one risk-free asset to smooth "
        "consumption across income states.\n\n"
        "The object is a stationary general equilibrium. Households choose an "
        "asset policy, firms demand capital, and the interest rate makes "
        "aggregate assets equal capital demand.\n\n"
        "The computation nests two fixed points. Value function iteration "
        "gives household saving at a candidate rate. Bisection updates the "
        "rate until the capital market clears."
    )

    report.add_equations(
        r"""
**Household.** Let $a_t\in[\underline a,\bar a]$ be beginning-of-period assets and
$z_t$ idiosyncratic labor efficiency. With prices $(r,w)$ and a no-borrowing
constraint $\underline a=0$, the household chooses next-period assets
$a_{t+1}=a'$ and consumes

$$c_t = (1+r)a_t + w z_t - a_{t+1},\qquad c_t>0.$$

Preferences are time-separable CRRA,

$$U_0 = \mathbb{E}_0\sum_{t=0}^{\infty}\beta^t u(c_t),\qquad
u(c)=\frac{c^{1-\sigma}}{1-\sigma},$$

with $\beta\in(0,1)$ and $\sigma>0$.

Log productivity is a Gaussian AR(1),

$$\log z_{t+1} = \rho\log z_t + \varepsilon_{t+1},\qquad
\varepsilon_{t+1}\sim\mathcal{N}(0,\sigma_\varepsilon^2),$$

approximated by an $N$-state Rouwenhorst chain. The grid is $\{z_j\}$, and
$P_{jk}=\Pr(z_{t+1}=z_k\mid z_t=z_j)$. The chain matches the AR(1) variance
and persistence, and is normalized so $\mathbb{E}[z]=1$.

The Bellman equation is

$$
V(a,z_j) = \max_{a'\in[\underline a,(1+r)a+wz_j)}
[u((1+r)a+wz_j-a') + \beta\sum_k P_{jk} V(a',z_k)],
$$

The solution gives asset policy $g_a(a,z_j)$. It also gives consumption policy
$c^{\ast}(a,z_j) = (1+r)a + w z_j - g_a(a,z_j)$. The borrowing constraint binds
whenever $g_a(a,z_j)=\underline a$.

**Stationary cross-section.** Given $g_a$ and $P$, the long-run distribution
$\mu$ over $(a,z)$ satisfies the operator equation

$$
\mu(a',z_k) = \sum_j P_{jk}\sum_{i:\,g_a(a_i,z_j)=a'} \mu(a_i,z_j),
$$

Here $i$ indexes nodes of the asset grid $\{a_i\}$. Aggregate household assets are
$K^s(r) = \sum_{i,j} a_i\,\mu(a_i,z_j)$.

**Firm.** Cobb-Douglas technology $Y = K^{\alpha} L^{1-\alpha}$ with capital
share $\alpha$ and depreciation $\delta$ delivers competitive factor prices

$$r(K) = \alpha(\tfrac{K}{L})^{\alpha-1}-\delta,\qquad
w(K) = (1-\alpha)(\tfrac{K}{L})^{\alpha}.$$

With aggregate efficient labor normalized to $L=1$, capital demand at $r$ is

$$K^d(r) = (\tfrac{r+\delta}{\alpha})^{1/(\alpha-1)}.$$

**Stationary equilibrium.** A stationary equilibrium contains price $r^{\ast}$,
wage $w^{\ast}$, policy $g_a$, and distribution $\mu$. The household problem is
solved at $(r^{\ast},w^{\ast})$. The distribution is invariant under
$(g_a,P)$. The capital market clears:

$$K^s(r^{\ast}) = \sum_{i,j} a_i\,\mu(a_i,z_j) = K^d(r^{\ast}).$$

A standard result is $r^{\ast}<1/\beta-1$. At or above that rate,
precautionary saving becomes unbounded. The economy has excess capital supply.
"""
    )

    report.add_model_setup(
        f"| Object | Value | Role |\n"
        f"|---|---:|---|\n"
        f"| Discount factor $\\beta$ | {beta} | Annual time preference |\n"
        f"| Impatience benchmark $1/\\beta-1$ | {impatience_rate:.4f} | Complete-markets ceiling on $r^{{\\ast}}$ |\n"
        f"| CRRA $\\sigma$ | {sigma_crra} | Curvature; controls precautionary motive |\n"
        f"| Capital share $\\alpha$ | {alpha} | Cobb-Douglas exponent on $K$ |\n"
        f"| Depreciation $\\delta$ | {delta} | Pinning $K^d(r)$ |\n"
        f"| Income persistence $\\rho$ | {rho} | AR(1) coefficient on $\\log z$ |\n"
        f"| Innovation s.d. $\\sigma_\\varepsilon$ | {sigma_eps} | AR(1) shock scale |\n"
        f"| Income states $N$ | {n_income} | Rouwenhorst nodes for $\\{{z_j\\}}$ |\n"
        f"| Asset bracket | $[{a_min:.0f},{a_max:.0f}]$ | $\\underline a$ at no-borrowing limit |\n"
        f"| Asset grid (coarse) | {n_asset} pts | Exponential, denser at $\\underline a$ |\n"
        f"| Capital-market tolerance | {tol_r:.0e} | Stop when $\\lvert K^s-K^d\\rvert/K^d$ falls below |\n"
        f"| Bracket-width tolerance | {tol_r_interval:.0e} | Backup stop on $r_H-r_L$ |\n"
        f"| VFI tolerance | {tol_vfi:.0e} | Sup-norm on $V$ |"
    )

    report.add_solution_method(
        "At a candidate $r$, firm first-order conditions imply $K^d(r)$ and "
        "$w(r)$. The household Bellman problem returns an asset policy. "
        "Forward iteration under that policy gives a stationary distribution. "
        "Aggregating assets gives household capital supply $K^s(r)$.\n\n"
        "Bisection compares $K^s(r)$ with $K^d(r)$. If households save too "
        "much, the rate is too high. If they save too little, the rate is too "
        f"low. The run stops after **{ge_iter}** bisection steps, with relative "
        f"gap **{abs(market_gap_rel):.2e}**.\n\n"
        "```text\n"
        "Algorithm: stationary Aiyagari equilibrium\n"
        "Inputs: primitives, asset grid, income chain, bracket [r_L, r_H]\n"
        "Output: r*, w*, K*, household policy, stationary distribution\n"
        "\n"
        "while not converged:\n"
        "    r = (r_L + r_H) / 2\n"
        "    compute K^d(r) and w(r) from firm first-order conditions\n"
        "    solve the household Bellman problem at (r, w)\n"
        "    iterate the joint distribution over (a, z) to stationarity\n"
        "    compute K^s(r) from the stationary distribution\n"
        "    if K^s(r) > K^d(r): set r_H = r\n"
        "    else: set r_L = r\n"
        "```\n\n"
        f"The final household VFI takes **{sol['iterations']}** iterations. "
        f"The sup-norm residual is below {tol_vfi:.0e}."
    )

    # =========================================================================
    # Figure 1: capital market (the headline picture; thumb is generated from this)
    # =========================================================================
    fig_capital, ax_capital = plt.subplots()
    r_demand = np.linspace(0.005, impatience_rate - 5e-4, 200)
    K_demand_curve = np.array([capital_demand(rv) for rv in r_demand])
    ax_capital.plot(K_demand_curve, r_demand, color="tab:blue", linewidth=2.0,
                    label="Firm demand $K^d(r)$")
    ax_capital.plot(Ks_curve, r_supply_grid, color="tab:red", linewidth=2.0,
                    marker="o", markersize=4,
                    label="Household supply $K^s(r)$")
    ax_capital.axhline(impatience_rate, color="0.45", linestyle=":", linewidth=1.0,
                       label="$1/\\beta - 1$")
    ax_capital.scatter([K_eq], [r_eq], s=120, marker="*", color="black", zorder=5,
                       label=f"Equilibrium ($K^{{\\ast}}={K_eq:.2f}$, "
                             f"$r^{{\\ast}}={r_eq:.4f}$)")
    ax_capital.set_xlabel("Aggregate capital $K$")
    ax_capital.set_ylabel("Interest rate $r$")
    ax_capital.set_title("Capital-Market Clearing")
    ax_capital.set_xlim(0.85 * min(Ks_curve.min(), K_demand_curve.min()),
                        1.05 * max(Ks_curve.max(), K_demand_curve.max(), K_eq))
    ax_capital.set_ylim(0.0, impatience_rate + 5e-3)
    ax_capital.legend(loc="upper right", fontsize=9)
    report.add_figure(
        "figures/capital-market.png",
        "Capital demand and household supply schedules with the stationary equilibrium",
        fig_capital,
        description=(
            "The firm schedule is analytic and slopes down. The household "
            "schedule solves the Bellman problem at each rate. The crossing "
            "is the stationary equilibrium. "
            f"At the calibration $r^{{\\ast}}={r_eq:.4f}$, roughly "
            f"{100*(impatience_rate-r_eq)/impatience_rate:.0f}% below the "
            "complete-markets benchmark."
        ),
    )

    # =========================================================================
    # Figure 2: value and asset policy (2-panel)
    # =========================================================================
    plot_states = [0, n_income // 4, n_income // 2, 3 * n_income // 4, n_income - 1]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, n_income))

    fig_pol, (ax_v, ax_g) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for j in plot_states:
        ax_v.plot(a_grid, sol["value"][:, j], color=cmap[j], linewidth=2.0,
                  label=f"$z_j={z_grid[j]:.2f}$")
    ax_v.set_xlabel("Assets $a$")
    ax_v.set_ylabel("$V(a, z_j)$")
    ax_v.set_title("Value function")
    ax_v.set_xlim(0, min(a_max, 30))
    ax_v.legend(loc="lower right", fontsize=8)

    for j in plot_states:
        ax_g.plot(a_grid, sol["asset_policy"][:, j], color=cmap[j], linewidth=2.0,
                  label=f"$z_j={z_grid[j]:.2f}$")
    ax_g.plot(a_grid, a_grid, color="0.4", linestyle="--", linewidth=0.9,
              label="$a'=a$")
    ax_g.set_xlabel("Current assets $a$")
    ax_g.set_ylabel("Next-period assets $a'$")
    ax_g.set_title("Asset policy $g_a(a, z_j)$")
    ax_g.set_xlim(0, min(a_max, 30))
    ax_g.set_ylim(0, min(a_max, 30))
    ax_g.legend(loc="upper left", fontsize=8)
    fig_pol.tight_layout()
    report.add_figure(
        "figures/savings-policy.png",
        "Equilibrium value function and asset policy",
        fig_pol,
        description=(
            "Value functions rise with assets and income. Asset policies show "
            "stronger saving after good income states. Near zero assets, the "
            "borrowing limit creates a visible kink in the policy. Each "
            "income line eventually crosses below the 45-degree line. That "
            "crossing gives a buffer-stock target at the equilibrium rate."
        ),
    )

    # =========================================================================
    # Figure 3: stationary wealth distribution + Lorenz curve
    # =========================================================================
    fig_dist, (ax_w, ax_l) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Bin onto a wider regular grid so grid-aligned spikes do not dominate the eye.
    a_max_plot = float(min(a_max, max(8.0, p90_wealth * 2.2, mean_wealth * 2.5)))
    n_bins = 50
    bin_edges = np.linspace(0.0, a_max_plot, n_bins + 1)
    bin_idx = np.clip(np.digitize(a_grid, bin_edges) - 1, 0, n_bins - 1)
    bin_mass = np.zeros(n_bins)
    np.add.at(bin_mass, bin_idx, asset_marginal)
    bin_mass = bin_mass / np.diff(bin_edges)
    ax_w.bar(bin_edges[:-1], bin_mass, width=np.diff(bin_edges), align="edge",
             color="steelblue", alpha=0.75, edgecolor="navy", linewidth=0.3)
    ax_w.axvline(mean_wealth, color="tab:red", linestyle="--", linewidth=1.4,
                 label=f"Mean = {mean_wealth:.2f}")
    ax_w.axvline(median_wealth, color="darkorange", linestyle=":", linewidth=1.4,
                 label=f"Median = {median_wealth:.2f}")
    ax_w.set_xlabel("Assets $a$")
    ax_w.set_ylabel("Density")
    ax_w.set_title("Stationary distribution of assets")
    ax_w.set_xlim(0, a_max_plot)
    ax_w.legend(loc="upper right", fontsize=9)

    pop = np.concatenate([[0.0], cum_pop])
    share = np.concatenate([[0.0], wealth_share])
    ax_l.plot(pop, share, color="tab:red", linewidth=2.0,
              label=f"Lorenz curve (Gini = {gini:.3f})")
    ax_l.plot([0, 1], [0, 1], color="0.4", linestyle="--", linewidth=0.9,
              label="Equality")
    ax_l.set_xlabel("Population share (sorted by assets)")
    ax_l.set_ylabel("Wealth share")
    ax_l.set_title("Lorenz curve")
    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(0, 1)
    ax_l.set_aspect("equal")
    ax_l.legend(loc="upper left", fontsize=9)
    fig_dist.tight_layout()
    report.add_figure(
        "figures/wealth-distribution.png",
        "Stationary wealth distribution and Lorenz curve at the equilibrium prices",
        fig_dist,
        description=(
            "The stationary distribution comes from the asset policy and "
            f"income chain at $r^{{\\ast}}={r_eq:.4f}$. Mean wealth "
            f"($\\mathbb{{E}}[a]={mean_wealth:.2f}$) exceeds median wealth "
            f"($\\tilde a={median_wealth:.2f}$). "
            f"${100*frac_constrained:.1f}\\%$ of households sit at the "
            "borrowing limit. The right tail comes from repeated high-income "
            f"draws. With no ex-ante heterogeneity, the run produces "
            f"Gini $G={gini:.3f}$."
        ),
    )

    # =========================================================================
    # Equilibrium summary table
    # =========================================================================
    summary = pd.DataFrame({
        "Variable": [
            "Interest rate $r^{\\ast}$",
            "Wage $w^{\\ast}$",
            "Aggregate capital $K^{\\ast}$",
            "Output $Y^{\\ast}$",
            "Capital-output ratio $K/Y$",
            "Mean wealth $\\mathbb{E}[a]$",
            "Median wealth $\\tilde a$",
            "P90 wealth",
            "Gini",
            "Mass at constraint",
            "Relative market-clearing gap",
        ],
        "Value": [
            f"{r_eq:.6f}",
            f"{w_eq:.4f}",
            f"{K_eq:.4f}",
            f"{Y_eq:.4f}",
            f"{K_over_Y:.4f}",
            f"{mean_wealth:.4f}",
            f"{median_wealth:.4f}",
            f"{p90_wealth:.4f}",
            f"{gini:.4f}",
            f"{frac_constrained:.4f}",
            f"{market_gap_rel:+.3e}",
        ],
    })
    report.add_table(
        "tables/equilibrium.csv",
        "Stationary equilibrium diagnostics",
        summary,
        description=(
            "The market-clearing gap is the bisection residual. It is "
            "numerical error, not a model object."
        ),
    )

    report.add_takeaway(
        "Precautionary saving turns a household policy into an aggregate "
        "capital supply curve. In this calibration, capital-market clearing "
        f"gives $r^{{\\ast}}={r_eq:.4f}$, below "
        f"$1/\\beta-1={impatience_rate:.4f}$. The lower rate is the price of "
        "incomplete insurance. The computation shows how VFI, a stationary "
        "distribution, and bisection close the model."
    )

    report.add_references([
        "Aiyagari, S. R. (1994). Uninsured Idiosyncratic Risk and Aggregate "
        "Saving. *Quarterly Journal of Economics*, 109(3), 659-684.",
        "Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic "
        "Theory*. MIT Press, 4th edition, Ch. 18.",
        "**See also.** The continuous-time analogue with HJB and KFE solvers "
        "is in [`heterogeneous-agents/aiyagari-hact/`](../../heterogeneous-agents/aiyagari-hact/), "
        "and the steady state here is reused as the household block of the "
        "HANK model in [`heterogeneous-agents/sequence-space-jacobian-hank/`](../../heterogeneous-agents/sequence-space-jacobian-hank/), "
        "where sequence-space Jacobians give the impulse responses to MIT shocks.",
    ])

    report.write("README.md")
    print(f"\nWrote README.md, {len(report._figures)} figures, "
          f"and {len(report._tables)} tables.")


if __name__ == "__main__":
    main()
