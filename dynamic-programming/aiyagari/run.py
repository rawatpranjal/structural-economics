#!/usr/bin/env python3
"""Aiyagari (1994) General Equilibrium Model with Heterogeneous Agents.

Solves for the stationary equilibrium of the Aiyagari model: households face
uninsurable idiosyncratic income risk and a borrowing constraint, and a
representative firm operates a Cobb-Douglas technology. The interest rate
adjusts to clear the capital market (aggregate savings = firm capital demand).

Reference: Aiyagari, S. R. (1994). "Uninsured Idiosyncratic Risk and
Aggregate Saving." Quarterly Journal of Economics, 109(3), 659-684.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.grids import exponential_grid
from lib.discretize import rouwenhorst
from lib.plotting import setup_style
from lib.output import ModelReport


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    # Preferences
    beta = 0.96          # Discount factor
    sigma_crra = 2.0     # CRRA risk aversion

    # Technology (Cobb-Douglas)
    alpha = 0.36         # Capital share
    delta = 0.08         # Depreciation rate

    # Income process: log(z') = rho * log(z) + eps, eps ~ N(0, sigma_eps^2)
    rho = 0.9            # Income persistence
    sigma_eps = 0.2      # Std dev of income innovation
    n_income = 7         # Income states (Rouwenhorst)

    # Asset grid
    n_asset = 200        # Asset grid points
    a_min = 0.0          # No-borrowing lower bound
    a_max = 50.0         # Maximum asset level

    # Equilibrium computation
    impatience_rate = 1.0 / beta - 1.0
    r_low = 0.01         # Lower bound for interest rate search
    r_high = impatience_rate - 0.001  # Upper bound, just below 1/beta - 1
    tol_r = 5e-4         # Relative tolerance for capital market clearing
    tol_r_interval = 1e-6  # Stop if bisection r interval is smaller than this
    max_iter_ge = 60     # Maximum GE iterations

    # VFI settings
    tol_vfi = 1e-6       # VFI convergence tolerance
    max_iter_vfi = 2000  # Maximum VFI iterations

    # =========================================================================
    # CRRA Utility
    # =========================================================================
    def u(c):
        """CRRA utility (vectorized, numpy)."""
        c_safe = np.maximum(c, 1e-15)
        if sigma_crra == 1.0:
            return np.log(c_safe)
        else:
            return c_safe ** (1 - sigma_crra) / (1 - sigma_crra)

    # =========================================================================
    # Income Process: Rouwenhorst discretization of AR(1) in logs
    # =========================================================================
    z_grid_jax, trans_jax, ergo_dist_jax = rouwenhorst(
        n=n_income, mu=0.0, sigma=sigma_eps, rho=rho
    )
    z_grid_log = np.array(z_grid_jax).flatten()   # log(z) values
    z_grid = np.exp(z_grid_log)                     # z values in levels
    trans = np.array(trans_jax)                      # (n_income, n_income)
    ergo_dist = np.array(ergo_dist_jax).flatten()

    # Normalize so mean income = 1 (aggregate labor supply L = 1)
    mean_z = np.dot(ergo_dist, z_grid)
    z_grid = z_grid / mean_z
    L_supply = 1.0  # Aggregate effective labor (normalized)

    print("Income grid (levels, normalized):", np.round(z_grid, 4))
    print("Ergodic distribution:", np.round(ergo_dist, 4))

    # =========================================================================
    # Asset Grid (exponential: denser near borrowing limit)
    # =========================================================================
    a_grid_jax = exponential_grid(a_min, a_max, n_asset, density=3.0)
    a_grid = np.array(a_grid_jax)

    # =========================================================================
    # Firm Problem
    # =========================================================================
    def firm_r(K):
        """Marginal product of capital minus depreciation."""
        return alpha * (K / L_supply) ** (alpha - 1) - delta

    def firm_w(K):
        """Marginal product of labor (wage)."""
        return (1 - alpha) * (K / L_supply) ** alpha

    def capital_demand(r):
        """Capital demand implied by interest rate r."""
        return L_supply * ((r + delta) / alpha) ** (1.0 / (alpha - 1.0))

    # =========================================================================
    # Solve Household Problem via VFI (given prices r, w)
    # =========================================================================
    def solve_household(r, w, V_init=None, verbose=False):
        """Solve the household consumption-savings problem for given (r, w).

        Returns:
            V: value function (n_asset, n_income)
            policy_a_idx: policy function indices (n_asset, n_income)
            policy_a: savings policy a'(a,z) (n_asset, n_income)
            policy_c: consumption policy c(a,z) (n_asset, n_income)
            info: dict with convergence info
        """
        # Initial guess
        if V_init is not None:
            V = V_init.copy()
        else:
            V = np.zeros((n_asset, n_income))
            for iz in range(n_income):
                for ia in range(n_asset):
                    coh = (1 + r) * a_grid[ia] + w * z_grid[iz]
                    V[ia, iz] = u(np.array([coh]))[0] / (1 - beta)

        for iteration in range(1, max_iter_vfi + 1):
            V_new = np.zeros((n_asset, n_income))
            policy_a_idx = np.zeros((n_asset, n_income), dtype=int)

            for iz in range(n_income):
                # Expected continuation value: E[V(a', z') | z]
                EV = V @ trans[iz, :]  # shape (n_asset,)

                # Vectorized over all current asset states
                cash_on_hand = (1 + r) * a_grid + w * z_grid[iz]  # (n_asset,)
                # consumption[ia, ia_next] = cash_on_hand[ia] - a_grid[ia_next]
                consumption = cash_on_hand[:, None] - a_grid[None, :]  # (n_asset, n_asset)
                feasible = consumption > 1e-10
                values = np.where(feasible, u(np.maximum(consumption, 1e-15)) + beta * EV[None, :], -1e20)
                V_new[:, iz] = np.max(values, axis=1)
                policy_a_idx[:, iz] = np.argmax(values, axis=1)

            error = np.max(np.abs(V_new - V))
            V = V_new

            if verbose and iteration % 100 == 0:
                print(f"    VFI iteration {iteration:4d}, error = {error:.2e}")

            if error < tol_vfi:
                if verbose:
                    print(f"    VFI converged in {iteration} iterations (error = {error:.2e})")
                break

        # Extract policy functions
        policy_a = np.zeros((n_asset, n_income))
        policy_c = np.zeros((n_asset, n_income))
        for iz in range(n_income):
            for ia in range(n_asset):
                cash_on_hand = (1 + r) * a_grid[ia] + w * z_grid[iz]
                policy_a[ia, iz] = a_grid[policy_a_idx[ia, iz]]
                policy_c[ia, iz] = cash_on_hand - policy_a[ia, iz]

        info = {"iterations": iteration, "converged": error < tol_vfi, "error": error}
        return V, policy_a_idx, policy_a, policy_c, info

    # =========================================================================
    # Stationary Distribution via Forward Iteration
    # =========================================================================
    def stationary_distribution(policy_a_idx):
        """Compute stationary distribution over (a, z) via forward iteration.

        Uses the policy function and transition matrix to iterate on the
        distribution until convergence.

        Returns:
            dist: (n_asset, n_income) stationary distribution
        """
        dist = np.ones((n_asset, n_income)) / (n_asset * n_income)
        tol_dist = 1e-10
        max_iter_dist = 5000

        for it in range(max_iter_dist):
            dist_new = np.zeros((n_asset, n_income))

            for iz in range(n_income):
                for ia in range(n_asset):
                    if dist[ia, iz] > 0:
                        ia_next = policy_a_idx[ia, iz]
                        for iz_next in range(n_income):
                            dist_new[ia_next, iz_next] += (
                                dist[ia, iz] * trans[iz, iz_next]
                            )

            error_dist = np.max(np.abs(dist_new - dist))
            dist = dist_new

            if error_dist < tol_dist:
                break

        return dist

    # =========================================================================
    # General Equilibrium: Bisection on r
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"General Equilibrium: Bisecting on r in [{r_low:.4f}, {r_high:.4f}]")
    print(f"{'='*60}")

    # Store results for capital supply/demand curves
    r_history = []
    Ks_history = []
    Kd_history = []

    V_warm = None  # Warm start for VFI

    for ge_iter in range(1, max_iter_ge + 1):
        r_trial = 0.5 * (r_low + r_high)
        w_trial = firm_w(capital_demand(r_trial))
        K_demand = capital_demand(r_trial)

        print(f"\nGE iteration {ge_iter}: r = {r_trial:.6f}, w = {w_trial:.4f}, K_demand = {K_demand:.4f}")

        # Solve household problem
        V, policy_a_idx, policy_a, policy_c, vfi_info = solve_household(
            r_trial, w_trial, V_init=V_warm, verbose=True
        )
        V_warm = V  # Use as warm start next iteration

        # Compute stationary distribution
        dist = stationary_distribution(policy_a_idx)

        # Aggregate capital supply = aggregate savings
        K_supply = np.sum(dist * a_grid[:, None])

        excess = K_supply - K_demand

        r_history.append(r_trial)
        Ks_history.append(K_supply)
        Kd_history.append(K_demand)

        print(f"  K_supply = {K_supply:.4f}, K_demand = {K_demand:.4f}, excess = {excess:.4f}")

        if abs(excess) / K_demand < tol_r:
            print(f"\n  *** Capital market clears at r = {r_trial:.6f} ***")
            break

        if excess > 0:
            # Too much saving => r is too high => lower r_high
            r_high = r_trial
        else:
            # Too little saving => r is too low => raise r_low
            r_low = r_trial

        if r_high - r_low < tol_r_interval:
            print(f"\n  *** Bisection interval exhausted (dr = {r_high - r_low:.2e}), "
                  f"using current midpoint r = {r_trial:.6f} ***")
            break

    # =========================================================================
    # Equilibrium Outcomes
    # =========================================================================
    r_eq = r_trial
    K_demand_eq = K_demand
    K_supply_eq = K_supply
    market_clearing_gap = K_supply_eq - K_demand_eq
    market_clearing_gap_pct = market_clearing_gap / K_demand_eq
    K_eq = K_demand_eq
    w_eq = firm_w(K_eq)
    Y_eq = K_eq ** alpha * L_supply ** (1 - alpha)

    # Wealth distribution statistics
    wealth_dist = np.sum(dist, axis=1)  # Marginal distribution over assets
    mean_wealth = K_supply_eq
    var_wealth = np.sum(wealth_dist * a_grid ** 2) - mean_wealth ** 2
    std_wealth = np.sqrt(max(var_wealth, 0))

    # Gini coefficient
    # Sort assets and compute Lorenz curve
    # Use the full (a, z) distribution flattened
    a_flat = np.repeat(a_grid, n_income)
    dist_flat = dist.flatten()

    # Sort by asset level
    sort_idx = np.argsort(a_flat)
    a_sorted = a_flat[sort_idx]
    dist_sorted = dist_flat[sort_idx]

    # Cumulative population and wealth shares
    cum_pop = np.cumsum(dist_sorted)
    cum_wealth = np.cumsum(dist_sorted * a_sorted)
    total_wealth = cum_wealth[-1]

    if total_wealth > 0:
        cum_wealth_share = cum_wealth / total_wealth
        # Gini = 1 - 2 * area under Lorenz curve
        gini = 1.0 - 2.0 * np.trapezoid(cum_wealth_share, cum_pop)
    else:
        gini = 0.0

    # Fraction at borrowing constraint
    frac_constrained = np.sum(dist[0, :])
    median_wealth = a_sorted[np.searchsorted(cum_pop, 0.5)]
    p90_wealth = a_sorted[np.searchsorted(cum_pop, 0.9)]
    if abs(market_clearing_gap_pct) < tol_r:
        ge_stop_reason = "the capital-market gap met the target tolerance"
    else:
        ge_stop_reason = (
            "the interest-rate bracket was exhausted; on this discrete asset grid the "
            "capital-supply schedule moves in small jumps"
        )

    print(f"\n{'='*60}")
    print(f"EQUILIBRIUM RESULTS")
    print(f"{'='*60}")
    print(f"Interest rate r:       {r_eq:.6f}")
    print(f"Wage w:                {w_eq:.4f}")
    print(f"Capital demand Kd:     {K_demand_eq:.4f}")
    print(f"Household assets Ks:   {K_supply_eq:.4f}")
    print(f"Market-clearing gap:   {market_clearing_gap:.4e}")
    print(f"Output Y:              {Y_eq:.4f}")
    print(f"K/Y ratio:             {K_eq/Y_eq:.4f}")
    print(f"Mean wealth:           {mean_wealth:.4f}")
    print(f"Std wealth:            {std_wealth:.4f}")
    print(f"Gini coefficient:      {gini:.4f}")
    print(f"Frac. at constraint:   {frac_constrained:.4f}")
    print(f"r vs 1/beta-1:         {r_eq:.6f} < {impatience_rate:.6f}")

    # =========================================================================
    # Generate Additional Data for Capital Supply/Demand Curves
    # =========================================================================
    # Compute capital demand curve analytically for a range of r
    r_curve = np.linspace(0.005, impatience_rate - 0.002, 50)
    Kd_curve = np.array([capital_demand(r_val) for r_val in r_curve])

    # For the supply curve, use the r values we evaluated during bisection
    # Sort for plotting
    sort_ge = np.argsort(r_history)
    r_sorted_ge = np.array(r_history)[sort_ge]
    Ks_sorted_ge = np.array(Ks_history)[sort_ge]

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Aiyagari Saving and Capital-Market Clearing",
        "A stationary incomplete-markets economy where household buffer stocks determine aggregate capital.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Aiyagari (1994) closes the income-risk savings problem in general equilibrium. "
        "Individual households still use assets as self-insurance, as in the "
        "[buffer-stock savings tutorial](../consumption-savings/), but the risk-free return "
        "is no longer imposed from outside the model. It is the price that makes aggregate "
        "household assets equal the firm's demand for productive capital.\n\n"
        "That feedback is the economic object of the tutorial. Persistent idiosyncratic "
        "risk makes households want precautionary buffers; the representative firm wants "
        "capital only up to the point where its marginal product justifies the rental rate. "
        "The stationary equilibrium is where those two schedules meet. If the household "
        "solve is the bottleneck, the [EGP tutorial](../../heterogeneous-agents/endogenous-grid-points/) "
        "shows the Euler-equation inversion behind faster incomplete-market solvers; this "
        "page keeps the general-equilibrium fixed point explicit."
    )

    report.add_equations(
        r"""
Let $a_t$ be beginning-of-period assets, $z_t$ idiosyncratic labor efficiency,
and $P_{jk}=\Pr(z_{t+1}=z_k\mid z_t=z_j)$ the Markov transition matrix. Given
prices $(r,w)$, a household chooses next-period assets $a_{t+1}=a'$ and consumes

$$c_t=(1+r)a_t+wz_t-a_{t+1}.$$

The no-borrowing constraint is $a_{t+1}\geq \underline a=0$. Preferences are

$$u(c)=\frac{c^{1-\sigma}}{1-\sigma},\qquad \sigma>0,\quad \sigma\neq 1,$$

and log efficiency follows

$$\log z_{t+1}=\rho\log z_t+\varepsilon_{t+1},\qquad
\varepsilon_{t+1}\sim N(0,\sigma_\varepsilon^2),$$

which is approximated by a Rouwenhorst chain. The household Bellman equation is

$$
V(a,z_j)=
\max_{\underline a\leq a'\leq \bar a,\ a'\leq (1+r)a+wz_j}
\left[
u((1+r)a+wz_j-a')+
\beta\sum_{k=1}^J P_{jk}V(a',z_k)
\right].
$$

The asset policy is $g_a(a,z_j)$. Given this policy, the stationary distribution
$\mu$ over asset-income states satisfies

$$
\mu(a_i,z_k)=
\sum_{j=1}^J\sum_{\ell:g_a(a_\ell,z_j)=a_i}\mu(a_\ell,z_j)P_{jk}.
$$

The firm has production $Y=K^\alpha L^{1-\alpha}$ and competitive factor prices

$$
r(K)=\alpha K^{\alpha-1}L^{1-\alpha}-\delta,\qquad
w(K)=(1-\alpha)K^\alpha L^{-\alpha}.
$$

With aggregate labor normalized to $L=1$, capital demand at interest rate $r$ is

$$
K^d(r)=\left(\frac{r+\delta}{\alpha}\right)^{1/(\alpha-1)}.
$$

An Aiyagari stationary equilibrium is a price $r^{*}$, wage $w^{*}$, household policy
$g_a$, and stationary distribution $\mu$ such that

$$
K^s(r^{*})=\sum_{i,j} a_i\mu(a_i,z_j)=K^d(r^{*}).
$$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\beta$ | {beta} | Discount factor |\n"
        f"| $1/\\beta-1$ | {impatience_rate:.4f} | Complete-markets impatience benchmark |\n"
        f"| $\\sigma$ | {sigma_crra} | CRRA risk aversion |\n"
        f"| $\\alpha$ | {alpha} | Capital share |\n"
        f"| $\\delta$ | {delta} | Depreciation rate |\n"
        f"| $\\rho$ | {rho} | Persistence of log income |\n"
        f"| $\\sigma_\\varepsilon$ | {sigma_eps} | Innovation standard deviation |\n"
        f"| $\\underline{{a}}$ | {a_min:.1f} | No-borrowing lower bound |\n"
        f"| $a \\in$ | [{a_min:.1f}, {a_max:.1f}] | Exponential asset grid support |\n"
        f"| Asset grid | {n_asset} points | Denser near $\\underline{{a}}$ |\n"
        f"| Income states | {n_income} | Rouwenhorst approximation to log income |\n"
        f"| Capital-market target | {tol_r:.0e} | Relative gap $\\lvert K^s-K^d\\rvert/K^d$ |\n"
        f"| Interest-rate bracket stop | {tol_r_interval:.0e} | Backup stopping rule for bisection |\n"
        f"| VFI tolerance | {tol_vfi:.0e} | Sup-norm value-function tolerance |"
    )

    report.add_solution_method(
        "There are two fixed points. The inner problem finds the optimal asset policy at a "
        "candidate price vector. The outer problem changes the interest rate until the "
        "stationary cross-section supplies the capital that firms demand.\n\n"
        "```text\n"
        "Algorithm 1: household block at candidate prices (r,w)\n"
        "Input: asset grid A, income states Z, transition matrix P, beta, utility u\n"
        "Output: value V(a,z), asset policy g_a(a,z), consumption policy c*(a,z)\n"
        "Initialize V_0(a_i,z_j) from consuming cash on hand forever\n"
        "repeat for n = 0, 1, 2, ...:\n"
        "    for each income state z_j:\n"
        "        C(a_i) = sum_k P_jk * V_n(a_i,z_k)\n"
        "        for each current asset a_i:\n"
        "            search over feasible next assets a_m in A\n"
        "            choose a_m maximizing u((1+r)*a_i + w*z_j - a_m) + beta*C(a_m)\n"
        "            store V_{n+1}(a_i,z_j) and g_a(a_i,z_j)\n"
        "    error = max_{i,j} |V_{n+1}(a_i,z_j) - V_n(a_i,z_j)|\n"
        "until error < epsilon_V\n"
        "set c*(a_i,z_j) = (1+r)*a_i + w*z_j - g_a(a_i,z_j)\n"
        "```\n\n"
        "```text\n"
        "Algorithm 2: stationary general equilibrium\n"
        "Input: interest-rate bracket [r_L,r_H], firm technology, household primitives\n"
        "Output: r*, w*, K, stationary distribution mu\n"
        "repeat:\n"
        "    r = (r_L + r_H) / 2\n"
        "    K^d(r) = ((r + delta) / alpha)^(1/(alpha - 1))\n"
        "    w = (1 - alpha) * (K^d(r))^alpha\n"
        "    solve Algorithm 1 at (r,w)\n"
        "    iterate mu forward under g_a and P until stationary\n"
        "    K^s(r) = sum_{i,j} a_i * mu(a_i,z_j)\n"
        "    if K^s(r) > K^d(r): set r_H = r\n"
        "    if K^s(r) < K^d(r): set r_L = r\n"
        "until |K^s(r) - K^d(r)| / K^d(r) < epsilon_K or r_H - r_L < epsilon_r\n"
        "```\n\n"
        f"The outer search stopped after **{ge_iter} iterations** because {ge_stop_reason}. "
        f"The final household "
        f"VFI took **{vfi_info['iterations']} iterations** and ended with sup-norm error "
        f"**{vfi_info['error']:.2e}**. The final signed relative capital-market gap is "
        f"**{market_clearing_gap_pct:.2e}**."
    )

    # --- Figure 1: Value Functions ---
    fig1, ax1 = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_income))
    # Plot a subset of income states for clarity
    plot_states = [0, n_income // 4, n_income // 2, 3 * n_income // 4, n_income - 1]
    for iz in plot_states:
        ax1.plot(a_grid, V[:, iz], color=colors[iz], linewidth=2,
                 label=f"$z = {z_grid[iz]:.2f}$")
    ax1.set_xlabel("Assets $a$")
    ax1.set_ylabel("$V(a, z)$")
    ax1.set_title("Lifetime Value at Equilibrium Prices")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, min(a_max, 30))
    report.add_results(
        "The value functions show the price of being liquidity constrained. Higher income "
        "states have higher lifetime value everywhere, but the vertical distance across "
        "income states is most visible at low assets. Once households hold larger buffers, "
        "current income matters less because the asset stock can absorb bad draws."
    )
    report.add_figure(
        "figures/value-functions.png",
        "Value functions by income state at equilibrium prices",
        fig1,
    )

    # --- Figure 2: Savings Policy Functions ---
    fig2, ax2 = plt.subplots()
    for iz in plot_states:
        ax2.plot(a_grid, policy_a[:, iz], color=colors[iz], linewidth=2,
                 label=f"$z = {z_grid[iz]:.2f}$")
    ax2.plot(a_grid, a_grid, "k:", linewidth=0.8, alpha=0.5, label="45-degree line")
    ax2.set_xlabel("Current assets $a$")
    ax2.set_ylabel("Next-period assets $a'$")
    ax2.set_title("Asset Policy at Equilibrium Prices")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, min(a_max, 30))
    ax2.set_ylim(0, min(a_max, 30))
    report.add_results(
        "The asset policy turns income histories into wealth dispersion. Points above the "
        "45-degree line mean accumulation; points below it mean drawdown. Low-income "
        "households use assets to smooth consumption and often move toward the constraint. "
        "High-income households save out of current resources, which is how persistent good "
        "income states feed the upper tail of the stationary wealth distribution."
    )
    report.add_figure(
        "figures/savings-policy.png",
        "Asset policy with the 45-degree no-change line",
        fig2,
    )

    # --- Figure 3: Stationary Wealth Distribution ---
    fig3, ax3 = plt.subplots()
    ax3.bar(a_grid, wealth_dist, width=np.diff(np.append(a_grid, a_grid[-1] + (a_grid[-1] - a_grid[-2]))),
            align="edge", color="steelblue", alpha=0.7, edgecolor="navy", linewidth=0.3)
    ax3.axvline(mean_wealth, color="red", linewidth=2, linestyle="--",
                label=f"Mean = {mean_wealth:.2f}")
    ax3.set_xlabel("Assets $a$")
    ax3.set_ylabel("Density")
    ax3.set_title("Stationary Wealth Distribution")
    ax3.legend(fontsize=10)
    ax3.set_xlim(0, min(a_max, max(8.0, p90_wealth * 2.2, mean_wealth * 2.0)))
    report.add_results(
        "The stationary distribution is not an assumption; it is implied by the policy and "
        f"the income Markov chain. Mean assets are **{mean_wealth:.2f}**, median assets are "
        f"**{median_wealth:.2f}**, and the 90th percentile is **{p90_wealth:.2f}**. The gap "
        "between the mean and the median is the visible aggregate counterpart of "
        f"self-insurance and persistent income risk. The wealth Gini is **{gini:.3f}**, "
        f"with **{frac_constrained:.1%}** of households exactly at the borrowing constraint."
    )
    report.add_figure(
        "figures/wealth-distribution.png",
        "Stationary wealth distribution implied by the equilibrium policy",
        fig3,
    )

    # --- Figure 4: Capital Supply vs Demand ---
    fig4, ax4 = plt.subplots()
    ax4.plot(Kd_curve, r_curve, "b-", linewidth=2, label="Capital demand $K^d(r)$")
    ax4.plot(Ks_sorted_ge, r_sorted_ge, "r-o", linewidth=2, markersize=5,
             label="Capital supply $K^s(r)$")
    ax4.axhline(r_eq, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax4.axhline(impatience_rate, color="black", linewidth=1.0, linestyle=":",
                alpha=0.8, label="$1/\\beta - 1$")
    ax4.axvline(K_eq, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax4.plot(K_eq, r_eq, "k*", markersize=15, zorder=5, label=f"Equilibrium ($K$={K_eq:.2f}, $r$={r_eq:.4f})")
    ax4.set_xlabel("Capital $K$")
    ax4.set_ylabel("Interest rate $r$")
    ax4.set_title("Capital Market Equilibrium")
    ax4.legend(fontsize=9)
    ax4.set_ylim(0, impatience_rate + 0.005)
    report.add_results(
        "The capital-market plot is the general-equilibrium step. Firm demand slopes down "
        "because the marginal product of capital falls with $K$. Household supply is shown "
        "at the rates visited by bisection, so it should be read as the fixed-point search, "
        "not as a separately smoothed object. The equilibrium rate is "
        f"**{r_eq:.4f}**, below the complete-markets impatience benchmark "
        f"**{impatience_rate:.4f}**, because uninsured income risk raises desired buffer "
        "assets."
    )
    report.add_figure(
        "figures/capital-market.png",
        "Capital demand and household capital supply in the equilibrium search",
        fig4,
    )

    # --- Table: Equilibrium Outcomes ---
    table_data = {
        "Variable": [
            "Interest rate $r$",
            "Wage $w$",
            "Firm capital demand $K^d$",
            "Aggregate household assets $K^s$",
            "Market-clearing gap $K^s-K^d$",
            "Relative gap $(K^s-K^d)/K^d$",
            "Output $Y$",
            "Capital-output ratio $K/Y$",
            "Gini coefficient",
            "Fraction at constraint",
            "$r$ vs $1/\\beta - 1$",
        ],
        "Value": [
            f"{r_eq:.6f}",
            f"{w_eq:.4f}",
            f"{K_demand_eq:.4f}",
            f"{K_supply_eq:.4f}",
            f"{market_clearing_gap:.4e}",
            f"{market_clearing_gap_pct:.4e}",
            f"{Y_eq:.4f}",
            f"{K_eq/Y_eq:.4f}",
            f"{gini:.4f}",
            f"{frac_constrained:.4f}",
            f"{r_eq:.6f} < {impatience_rate:.6f}",
        ],
    }
    df = pd.DataFrame(table_data)
    report.add_table(
        "tables/equilibrium.csv",
        "Stationary equilibrium diagnostics",
        df,
        description=(
            "The table separates firm capital demand from household asset supply because the "
            "computed equilibrium is numerical. The reported gap is the remaining market-"
            "clearing error, while the interest-rate comparison and the distributional "
            "statistics are the main economic diagnostics."
        ),
    )

    report.add_takeaway(
        "Incomplete markets turn a household precautionary motive into an aggregate price "
        "effect. With no insurance against persistent income risk, households want buffer "
        "assets; in equilibrium those assets are the economy's capital "
        f"stock. The result is $r^{{*}}={r_eq:.4f}$, below $1/\\beta-1={impatience_rate:.4f}$, "
        "and a right-skewed wealth distribution generated without ex ante heterogeneity. "
        "Faster algorithms can change the cost of the computation, but they do not change "
        "the fixed point: solve household policies, find the stationary distribution, and "
        "clear the capital market."
    )

    report.add_references([
        "Aiyagari, S. R. (1994). Uninsured Idiosyncratic Risk and Aggregate Saving. "
        "*Quarterly Journal of Economics*, 109(3), 659-684.",
        "Huggett, M. (1993). The Risk-Free Rate in Heterogeneous-Agent Incomplete-Insurance "
        "Economies. *Journal of Economic Dynamics and Control*, 17(5-6), 953-969.",
        "Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. "
        "MIT Press, 4th edition, Ch. 18.",
        "Kaplan, G., Moll, B., and Violante, G. L. (2018). Monetary Policy According to HANK. "
        "*American Economic Review*, 108(3), 697-743.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
