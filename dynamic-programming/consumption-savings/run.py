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

    mpc_mid = np.gradient(policy_c[:, median_z_idx], a_grid)
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

    report = ModelReport(
        "Income Risk and Buffer-Stock Saving",
        "A partial-equilibrium savings problem with persistent idiosyncratic income.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Households often face income paths they cannot insure. A worker with low "
        "assets must absorb a bad income draw through current consumption.\n\n"
        "The object is a household savings rule. The state is assets $a$ and income "
        "$z$. The control is next-period assets $a'$ under a no-borrowing limit.\n\n"
        "Income is persistent, so today's choice changes future cash-on-hand risk. "
        "Value function iteration solves the Bellman equation on an asset grid and "
        "a finite Markov chain for income."
    )

    report.add_equations(
        r"""
Let $a_t$ be beginning-of-period assets, $z_t$ labor income, and
$R=1+r$ the gross risk-free return. The household chooses next-period assets
$a_{t+1}=a'$ and consumes the residual

$$c_t = R a_t + z_t - a_{t+1}.$$

Assets must respect the no-borrowing constraint:

$$a_{t+1}\geq \underline a = 0,$$

The numerical problem also uses an upper grid bound $\bar a$. Utility is CRRA:

$$u(c)=\frac{c^{1-\sigma}}{1-\sigma}, \qquad \sigma>0,\quad \sigma\neq 1.$$

Log income follows:

$$\log z_{t+1}=\rho \log z_t+\varepsilon_{t+1},\qquad
\varepsilon_{t+1}\sim N(0,\sigma_\varepsilon^2),$$

It is approximated by $J$ income states $z_1,\ldots,z_J$ with transition matrix
$P$. Here $P_{jk}=\Pr(z_{t+1}=z_k\mid z_t=z_j)$. The Bellman equation is

$$
V(a,z_j)=
\max_{\underline a\leq a'\leq \bar a,\ a'\leq R a+z_j}
\left[
u(Ra+z_j-a')+
\beta\sum_{k=1}^J P_{jk}V(a',z_k)
\right].
$$

Here $\beta\in(0,1)$ is the discount factor. The asset policy is $g_a(a,z)=a'$. The consumption policy is
$c^{\ast}(a,z)=Ra+z-g_a(a,z)$. At an interior choice, the Euler equation is:

$$u'(c_t)=\beta R\,\mathbb{E}_t[u'(c_{t+1})],$$

When the constraint binds, $a_{t+1}=0$ and the Euler inequality holds:

$$u'(c_t)\geq \beta R\,\mathbb{E}_t[u'(c_{t+1})].$$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\beta$ | {beta} | Discount factor |\n"
        f"| $r$ | {r} | Exogenous risk-free interest rate |\n"
        f"| $R$ | {gross_return:.2f} | Gross return on assets |\n"
        f"| $\\beta R$ | {beta * gross_return:.4f} | Impatience margin; below one here |\n"
        f"| $\\sigma$ | {sigma_crra} | CRRA risk aversion |\n"
        f"| $\\rho$ | {rho} | Persistence of log income |\n"
        f"| $\\sigma_\\varepsilon$ | {sigma_eps} | Innovation standard deviation |\n"
        f"| $\\underline{{a}}$ | {borrowing_limit:.1f} | No-borrowing lower bound |\n"
        f"| $a \\in$ | [{a_min:.1f}, {a_max:.1f}] | Asset grid support |\n"
        f"| Asset state grid | {n_asset} points | Exponential spacing near $\\underline{{a}}$ |\n"
        f"| Next-asset choice grid | {n_choice} points | Candidate $a'$ values in each Bellman update |\n"
        f"| Refined diagnostic grid | {n_asset_refined} states, {n_choice_refined} choices | Held-out check for the median income state |\n"
        f"| Income states | {n_income} | Rouwenhorst approximation to log income |\n"
        f"| Simulation panel | {n_panel_agents} agents, 400 periods | Used only to illustrate the induced asset distribution |"
    )

    report.add_solution_method(
        "The value function is stored on a grid for assets and income. Income uses "
        "a five-point Rouwenhorst chain for $\\log z$.\n\n"
        "The Bellman operator is\n\n"
        "$$(TV)(a,z_j)=\\max_{0\\leq a'\\leq Ra+z_j}\\left[u(Ra+z_j-a')+\\beta\\sum_{k=1}^J "
        "P_{jk}V(a',z_k)\\right],$$\n\n"
        "a $\\beta$-contraction on bounded functions of $(a,z)$ (the grid upper bound "
        "$\\bar{a}$ is always slack at an interior solution). For each income "
        "state, the code computes expected continuation value on the asset grid. "
        "It interpolates that value onto a denser grid for $a'$.\n\n"
        "At each $(a,z)$, infeasible choices with $a'>Ra+z$ receive value "
        "$-\\infty$. The algorithm picks the best $a'$ and repeats until the "
        "sup-norm change is below tolerance.\n\n"
        "```text\n"
        "Algorithm  Income-fluctuation VFI\n"
        "Inputs   asset state grid A = {a_i}, asset choice grid G = {g_l},\n"
        "           income grid Z = {z_j}, transition P with P_{jk} = Pr(z' = z_k | z = z_j),\n"
        "           primitives (beta, R, sigma), utility u, tolerance epsilon\n"
        "Outputs  V*(a_i, z_j), asset policy g_a(a_i, z_j),\n"
        "           consumption policy c*(a_i, z_j) = R a_i + z_j - g_a(a_i, z_j)\n"
        "\n"
        "Initialise V_0(a_i, z_j) <- u(R a_i + z_j) / (1 - beta)        # eat-cash-on-hand guess\n"
        "for n = 0, 1, 2, ...:\n"
        "    for each income state z_j:\n"
        "        EV(a_i) <- sum_k P_{jk} * V_n(a_i, z_k)                # expected continuation on A\n"
        "        EV_hat(g_l) <- interp(EV from A to G)                  # off-state continuation on G\n"
        "        for each asset state a_i:\n"
        "            feasible(g_l) := { g_l <= R a_i + z_j }            # respects no-borrowing\n"
        "            obj(g_l) <- u(R a_i + z_j - g_l) + beta * EV_hat(g_l)\n"
        "            g_a(a_i, z_j) <- argmax_{feasible} obj\n"
        "            V_{n+1}(a_i, z_j) <- max obj\n"
        "    err <- max_{i,j} | V_{n+1}(a_i, z_j) - V_n(a_i, z_j) |\n"
        "    stop when err < epsilon\n"
        "```\n\n"
        f"The main grid converges in **{solution['iterations']} iterations** to "
        f"sup-norm residual **{solution['error']:.2e}**. A refined grid repeats the "
        f"same solve with {n_asset_refined} state points and {n_choice_refined} "
        "choice points. Its median-income policy is plotted below as a diagnostic."
    )

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_income))

    fig1, ax1 = plt.subplots()
    for iz in range(n_income):
        ax1.plot(a_grid, value[:, iz], color=colors[iz], linewidth=2, label=f"$z={z_grid[iz]:.3f}$")
    ax1.set_xlabel("Assets $a$")
    ax1.set_ylabel("$V(a,z)$")
    ax1.set_title("Value by Income State")
    ax1.legend(fontsize=9)
    report.add_results(
        "$V(a,z_j)$ rises with assets and income. Near $\\underline{a}=0$, income "
        "states have large value gaps. Low assets give the household little "
        "insurance against a bad draw."
    )
    report.add_figure(
        "figures/value-functions.png",
        "Value functions by income state",
        fig1,
    )

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
    report.add_results(
        "Consumption rises with assets and is steepest near the borrowing limit. "
        f"For the median income state, average MPC is **{low_asset_mpc:.2f}** near "
        f"zero assets and **{high_asset_mpc:.2f}** near the top. The fall measures "
        "buffer-stock saving. An extra dollar is mostly consumed when assets are "
        "scarce. It is mostly saved after the buffer is large.\n\n"
        "The dashed line is the refined-grid median-income policy. Its maximum gap "
        f"from the main grid is **{max_refined_gap_mid:.2e}**."
    )
    report.add_figure(
        "figures/consumption-policy.png",
        "Consumption policy with refined-grid benchmark on the median income state",
        fig2,
    )

    fig3, ax3 = plt.subplots()
    for iz in range(n_income):
        ax3.plot(a_grid, savings_policy[:, iz], color=colors[iz], linewidth=2, label=f"$z={z_grid[iz]:.3f}$")
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax3.set_xlabel("Assets $a$")
    ax3.set_ylabel("Net saving $g_a(a,z)-a$")
    ax3.set_title("Net Saving Policy")
    ax3.legend(fontsize=9)
    report.add_results(
        "Net saving $g_a(a,z_j)-a$ shows when the household builds the buffer. High "
        "income raises saving, especially close to the borrowing limit. Low income "
        "leads to dissaving until the constraint stops it. The zero crossing is the "
        "buffer-stock target for the median income state."
    )
    report.add_figure(
        "figures/savings-policy.png",
        "Net saving by asset and income state",
        fig3,
    )

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
    report.add_results(
        "Forward simulation applies the policy after each income draw. Five agents "
        "start with identical median income. Persistent shocks spread their assets "
        "over time. Runs of bad income push assets back toward the constraint.\n\n"
        "A panel of 3,000 agents shows the cross-section after 400 periods. Median "
        f"wealth is **{median_assets:.2f}**, and the 90th percentile is "
        f"**{p90_assets:.2f}**. About **{constraint_share:.1%}** of agents sit near "
        "$\\underline{a}$. The pile-up at zero and the right tail come from the "
        "policy and persistent risk."
    )
    report.add_figure(
        "figures/simulated-paths.png",
        "Simulated asset paths and the induced cross-sectional asset distribution",
        fig4,
    )

    report.add_takeaway(
        "Persistent income risk and no borrowing make saving state-contingent. "
        "Value function iteration turns the recursive choice into a policy on "
        "the asset-income grid. "
        "The computed policy has high MPCs near zero assets, positive saving after "
        "high income, and a finite buffer-stock target."
    )

    report.add_references([
        "Carroll, C. D. (1997). Buffer-Stock Saving and the Life Cycle/Permanent Income "
        "Hypothesis. *Quarterly Journal of Economics*, 112(1), 1-55.",
        "Deaton, A. (1991). Saving and Liquidity Constraints. *Econometrica*, 59(5), 1221-1248.",
        "Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. "
        "MIT Press, 4th edition, Ch. 18.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
