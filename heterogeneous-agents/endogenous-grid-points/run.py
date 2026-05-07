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
from lib.output import ModelReport
from lib.plotting import setup_style


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
    mpclim = gross_return * (beta_r ** (-1.0 / gamma)) - 1.0

    print("\nStationary cross section from simulation:")
    print(f"  Mean assets:              {mean_assets:.3f}")
    print(f"  Mean consumption:         {mean_consumption:.3f}")
    print(f"  Gini (wealth):            {gini_wealth:.3f}")
    print(f"  Average MPC, large shock: {mean_mpc_large:.3f}")
    print(f"  Fraction constrained:     {frac_constrained:.1f}%")
    print(f"  Theoretical MPC limit:    {mpclim:.4f}")
    print(f"  Consumption gap vs fine grid on a <= {accuracy_asset_max:g}: {consumption_gap:.2e}")
    print(f"  Savings gap vs fine grid on a <= {accuracy_asset_max:g}: {savings_gap:.2e}")

    setup_style()

    report = ModelReport(
        "Buffer-Stock Saving by Endogenous Grid Points",
        "Euler-equation inversion for a partial-equilibrium income-risk household problem.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "An impatient household with CRRA preferences faces IID labor-income risk "
        "and a no-borrowing constraint $\\underline a = 0$. The economic content is "
        "the buffer-stock logic of Deaton (1991) and Carroll (1997): assets are held "
        "purely to self-insure against bad income draws, and the constraint binds "
        "with strictly positive probability, so the Euler equation holds with "
        "complementary slackness. The "
        "[buffer-stock VFI tutorial](../../dynamic-programming/consumption-savings/) "
        "solves the persistent-income version by grid maximization over $a'$. The "
        "exercise here is to compute the same kind of policy without that inner "
        "maximization.\n\n"
        "The trick is Carroll's (2006) endogenous-grid trade. VFI iterates on "
        "$V(a,y)$ by asking, at each current $a$, which $a'$ delivers the highest "
        "$u(c) + \\beta\\,\\mathbb{E}V(a',y')$. EGP fixes a grid for *next-period* "
        "assets $a'$ instead, evaluates the Euler equation pointwise to recover the "
        "consumption that rationalizes each $a'$, and reads the implied current "
        "asset level off the budget identity. The map $a'\\mapsto a$ is then "
        "inverted by interpolation to deliver $g_a(a,y)$ on the original grid. No "
        "argmax, no gradient, just one expectation, one inverse marginal utility, "
        "and one univariate interpolation per income state per iteration.\n\n"
        "The reversal pays off whenever the household block is solved repeatedly "
        "inside an outer loop. The [Aiyagari tutorial](../../dynamic-programming/aiyagari/) "
        "bisects on $r$ around the household problem; "
        "[Huggett](../huggett-incomplete-markets/) does the same for a bond economy "
        "in continuous time. The neighbouring "
        "[envelope-equation iteration](../envelope-equation-iteration/) tutorial keeps "
        "the same Euler discipline but iterates on $W_a(a)$ rather than the consumption "
        "policy itself, and shows that grid VFI, EGP, and EEI all coincide on the "
        "buffer-stock policy at the resolution used here."
    )

    report.add_equations(
        r"""
The household enters the period with assets $a$ and observes income $y_j$ drawn
IID from $\{y_1,\dots,y_{n_y}\}$ with probabilities $\pi_j$. With gross return
$R=1+r$, it chooses next-period assets $a'=g(a,y_j)$, consumes the residual,
and faces a non-borrowing constraint:

$$
V(a,y_j) = \max_{a'\geq \underline a}
  [\,u(R a + y_j - a') + \beta\,\sum_{\ell=1}^{n_y}\pi_\ell\, V(a',y_\ell)\,],
\qquad c(a,y_j) = R a + y_j - g(a,y_j).
$$

Because income is IID, the continuation $\mathbb{E}V(a',y')$ depends only on
$a'$, which is what makes EGP especially clean here. Preferences are CRRA, so
the marginal utility map and its analytic inverse are

$$
u'(c) = c^{-\gamma}, \qquad (u')^{-1}(\mu) = \mu^{-1/\gamma}.
$$

At an interior optimum the Euler equation equates today's marginal utility
with the discounted marginal benefit of saving,

$$
\underbrace{u'(c(a,y_j))}_{\text{cost of saving today}}
= \beta R\,
\underbrace{\sum_{\ell=1}^{n_y}\pi_\ell\,u'\!(c(g(a,y_j),y_\ell))}_{\text{expected marginal utility tomorrow}}.
$$

When the borrowing limit binds, $g(a,y_j)=\underline a$ and the Euler condition
holds as an inequality,
$u'(c(a,y_j)) \geq \beta R \sum_\ell \pi_\ell u'(c(\underline a,y_\ell))$.
This Kuhn-Tucker margin is what generates large MPCs at low wealth: a marginal
dollar of cash relaxes a slack constraint dollar-for-dollar, so $\partial c/\partial a$
can be close to $R$ rather than the small perfect-foresight value
$1-(\beta R)^{1/\gamma}/R$.
"""
    )

    report.add_model_setup(
        f"| Object | Value | Role |\n"
        f"|---|---:|---|\n"
        f"| CRRA $\\gamma$ | {gamma:.1f} | Curvature; sets the strength of precautionary motive and shapes MPCs |\n"
        f"| Discount factor $\\beta$ | {beta:.2f} | Annual time preference |\n"
        f"| Net rate $r$ | {r:.2f} | Exogenous risk-free return |\n"
        f"| Patience-return product $\\beta R$ | {beta_r:.4f} | $<1$ rules out the unbounded-saving target of Carroll (1997) |\n"
        f"| Income mean $\\mu_y$ | {mean_income:.1f} | Normalization |\n"
        f"| Income s.d. $\\sigma_y$ | {sd_income:.1f} | Width of the IID labor-income shock |\n"
        f"| Income states $n_y$ | {n_income} | Width-fitted equal-spaced normal grid |\n"
        f"| Borrowing limit $\\underline a$ | {borrowing_limit:.1f} | Hard zero; binds with positive mass |\n"
        f"| Upper grid bound $\\bar a$ | {asset_max:.1f} | Set wide enough to contain the simulated tail |\n"
        f"| EGP asset grid | {n_asset} pts | Exponential, denser at $\\underline a$ |\n"
        f"| Audit grid | {n_asset_refined} pts | Fine-grid reference for the discretization check |\n"
        f"| Convergence tolerance | {tol:.0e} | Sup-norm on consumption iterates |\n"
        f"| Simulation | {n_agents:,} households, {periods} periods | Forward-iterated cross section under $g_a$ |"
    )

    report.add_solution_method(
        rf"""
**The key trade.** VFI on this problem maximizes $u((1+r)a+y_j-a')+\beta\,
\mathbb{{E}} V(a',y')$ over $a'$ at every state, paying a one-dimensional search
per grid point per iteration. EGP holds the grid $\{{a_i'\}}_{{i=1}}^{{N_a}}$ fixed
in the *next-period* assets, evaluates the Euler equation pointwise to recover
the consumption that is consistent with stepping to each $a_i'$, and reads the
implied current asset off the budget line:

$$
c_i = (u')^{{-1}}\!(\beta R \sum_{{\ell}} \pi_\ell\, u'(c_n(a_i', y_\ell))),
\qquad
a^{{\text{{endo}}}}_{{ij}} = \frac{{c_i + a_i' - y_j}}{{R}}.
$$

Because $c$ is strictly increasing in cash on hand and $u'$ is strictly
decreasing, the map $a_i' \mapsto a^{{\text{{endo}}}}_{{ij}}$ is monotone for each
$y_j$. Inverting it is therefore a single sorted interpolation onto the
exogenous grid $A$. The borrowing constraint enters as a left-tail boundary
correction: any $a < a^{{\text{{endo}}}}_{{1j}}$ cannot rationalize an interior
saving choice given $y_j$, so the policy is pinned at $g_a = \underline a$.

```text
Algorithm: EGP for IID-income buffer-stock saving
Inputs    grid {{a_i'}} (also serves as the exogenous current-asset grid),
          income chain ({{y_j}}, {{pi_j}}), primitives (beta, R, gamma),
          borrowing limit a_min, tolerance eps
Output    consumption policy c(a, y), saving policy g(a, y)

Initialise c_0(a_i, y_j) = (R-1) a_i + y_j        # consume current resources
repeat n = 0, 1, 2, ...
    # 1. Euler inversion at each candidate next asset a_i'
    M_i  = sum_l pi_l * u'(c_n(a_i', y_l))         # expected MU tomorrow
    c_i  = (u')^{{-1}}(beta R M_i)                  # consumption today

    for each income state y_j:
        # 2. Endogenous current asset
        a^endo_{{i,j}} = (c_i + a_i' - y_j) / R

        # 3. Invert by interpolation onto the exogenous grid A = {{a_i}}
        g_{{n+1}}(a_i, y_j) = interp(a_i; a^endo_{{:,j}}, a'_:)

        # 4. Constrained branch
        for each a_i <= a^endo_{{1,j}}:
            g_{{n+1}}(a_i, y_j) = a_min

        c_{{n+1}}(a_i, y_j) = R a_i + y_j - g_{{n+1}}(a_i, y_j)

    err = max_{{i,j}} |c_{{n+1}}(a_i, y_j) - c_n(a_i, y_j)|
until err < eps
```

Three observations help in practice. First, EGP inherits the geometric
contraction rate of the underlying Bellman operator, so iteration counts scale
with $\beta$, not with $N_a$. Second, the interpolation is over a sorted
sequence; using `np.interp` is fine and the extrapolation branch on the right
end matters only if the grid bound $\bar a$ is set aggressively low. Third,
when income is persistent (not the case here) the endogenous-current-asset
grid depends on $y_j$, and the inversion has to be done income state by income
state — the IID simplification used in this tutorial is a clean expository
benchmark, not a structural assumption.

**Convergence and accuracy.** The {n_asset}-point grid converged in
**{int(solution["iterations"])} EGP iterations** with a consumption sup-norm
residual of {float(solution["error"]):.2e}. To audit the discretization, the
same EGP solve was rerun on a {n_asset_refined}-point grid at the identical
calibration and the two policies were compared on $a \leq {accuracy_asset_max:g}$,
the asset range that holds essentially all of the simulated mass. The maximum
consumption-policy gap is {consumption_gap:.2e} and the next-asset gap is
{savings_gap:.2e}; both are pure grid-and-interpolation wedges with no
economic content. The fine grid is not used in the simulation — it appears
only as the dashed reference in the policy plots and as the diagnostic row in
the summary table.
"""
    )

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

    report.add_figure(
        "figures/consumption-policy.png",
        "Consumption policy with fine-grid EGP reference",
        fig1,
        description=(
            "The first figure shows the EGP consumption policy at five income "
            "states (the two extreme states bolded, the three interior states in "
            "grey) with the dashed fine-grid reference overlaid for the lowest "
            "and highest $y_j$. The shape is the same buffer-stock policy that "
            "VFI delivers in the [persistent-income tutorial]"
            "(../../dynamic-programming/consumption-savings/): concave, increasing "
            "in $a$, and shifted vertically by income because IID $y_j$ enters "
            "cash on hand directly. Slopes near $\\underline a$ are close to the "
            "45-degree reference $c=Ra+y_j$, the certainty-equivalent rule for a "
            "constrained agent who consumes everything; far from the constraint "
            "the slope falls toward the perfect-foresight limit "
            f"$\\kappa^{{\\ast}} \\approx {mpclim:.3f}$ derived from "
            "$c_{t+1}/c_t=(\\beta R)^{1/\\gamma}$. The coarse and fine-grid "
            "policies are visually indistinguishable on the plotted range, which "
            "is the discretization audit."
        ),
    )

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

    report.add_figure(
        "figures/savings-policy.png",
        "Net saving policy with fine-grid EGP reference",
        fig2,
        description=(
            "Net saving $g_a(a,y_j)-a$ separates income states more cleanly than "
            "consumption does. After a low income draw the household decumulates "
            "to consume more than $Ra+y_j$ for $a$ above the constraint, and "
            "rolls onto $g_a=\\underline a$ once cash on hand can no longer "
            "support an interior Euler-equation choice — that is the discrete "
            "kink at the left end of the lowest-income curve. A high draw flips "
            "the sign and rebuilds the buffer. The zero crossings are not steady "
            "states: with IID income, the household keeps cycling across asset "
            "states as draws arrive, and the simulated cross section averages "
            "over those cycles."
        ),
    )

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

    report.add_figure(
        "figures/endogenous-grid.png",
        "Endogenous current asset grid for the low income state",
        fig3,
        description=(
            "This third figure makes the EGP construction visible. For each "
            "$a_i'$ on the exogenous grid, the Euler inversion fixes "
            "$c_i = (u')^{-1}(\\beta R\\,\\mathbb{E}u'(c_n(a_i',y')))$, and the "
            "budget identity $a^{\\mathrm{endo}} = (c_i + a_i' - y_j)/R$ then "
            "pins down the current asset level that would have rationalized "
            "stepping to $a_i'$ after observing the lowest income draw "
            f"$y_1={income_grid[low]:.2f}$. The 45-degree line is the static "
            "no-saving rule $a^{\\mathrm{endo}}=a'$; the policy curve sits "
            "above it because the household with the lowest current income "
            "wants to draw down assets, so a given $a'$ requires a larger "
            "current $a$ to finance. The first endogenous point "
            f"$a^{{\\mathrm{{endo}}}}_{{1,1}}={endogenous_assets[0, low]:.3f}$ "
            "is the kink threshold: any current $a$ below it would force "
            "negative interior consumption, so the borrowing constraint "
            "supplies $g_a=\\underline a$ on that left tail."
        ),
    )

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

    report.add_figure(
        "figures/wealth-distribution.png",
        "Simulated terminal wealth distribution",
        fig4,
        description=(
            f"Forward-iterating $g_a$ for {periods} periods on {n_agents:,} "
            "households gives the cross section in the fourth figure. The "
            f"distribution is right-skewed with mean $\\bar a={mean_assets:.2f}$ "
            "and a small mass exactly at the constraint "
            f"({frac_constrained:.1f}\\% of agents); the spike at zero is the "
            "Kuhn-Tucker margin showing up in the marginal distribution. The "
            "scale is modest because income is IID — there is no persistence to "
            "amplify good histories — and because $\\beta R<1$ rules out a "
            "drifting asset target. Replacing IID income with the persistent "
            "[Rouwenhorst chain](../../dynamic-programming/shock-discretization/) "
            "and closing the model with capital-market clearing produces the "
            "much wider Aiyagari cross section in the "
            "[Aiyagari tutorial](../../dynamic-programming/aiyagari/)."
        ),
    )

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
                label=f"Perfect-foresight limit = {mpclim:.3f}")
    ax5.axvline(mean_mpc_large, color="darkorange", linestyle="--", linewidth=1.5,
                label=f"Mean = {mean_mpc_large:.3f}")
    ax5.set_xlabel("MPC out of a 0.10 transfer")
    ax5.set_ylabel("Density")
    ax5.set_title("Marginal Propensity to Consume")
    ax5.set_xlim(0, 1.05)
    ax5.legend()

    report.add_figure(
        "figures/mpc-distribution.png",
        "Distribution of marginal propensities to consume",
        fig5,
        description=(
            "The fifth figure plots the cross-sectional distribution of MPCs "
            "out of a small transfer of 0.10 — about ten percent of mean income. "
            f"The average MPC is {mean_mpc_large:.3f}, an order of magnitude above "
            f"the perfect-foresight limit $\\kappa^{{\\ast}}\\approx{mpclim:.3f}$ "
            "marked by the dotted line. The high values come from constrained or "
            "near-constrained households, for whom an extra dollar of cash is "
            "spent dollar-for-dollar; the right tail near 1 is exactly the "
            "Kuhn-Tucker margin from the equations section made empirical. The "
            "low-MPC mode near $\\kappa^{\\ast}$ is the wealth-rich subpopulation "
            "for whom the constraint is slack and the Euler equation pins the "
            "consumption response. This bimodality is the proximate reason "
            "heterogeneous-agent models can deliver aggregate consumption "
            "responses to fiscal transfers far above what a representative-agent "
            "PIH model implies — see the discussion in Kaplan and Violante (2022)."
        ),
    )

    table_data = {
        "Statistic": [
            "Mean assets",
            "Mean consumption",
            "Wealth Gini",
            "Average MPC, 0.10 transfer",
            "Average local MPC",
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
    report.add_table(
        "tables/summary-statistics.csv",
        "Simulation and Accuracy Summary",
        df,
        description=(
            "The summary table separates the economic outputs of the cross "
            "section (mean wealth, Gini, average MPCs, mass at the constraint) "
            "from the discretization diagnostics (consumption and savings gaps "
            "against the fine-grid solve). The Gini and the high average MPC "
            "are model results that depend on $\\gamma$, $\\beta R$, and the "
            "income process; the policy gaps are pure numerical wedges that "
            "shrink toward zero as $N_a\\to\\infty$."
        ),
    )

    report.add_takeaway(
        "Carroll's grid reversal is a workhorse precisely because it is not a new "
        "model. The same buffer-stock policy that VFI computes by maximizing over "
        f"$a'$ at every state falls out of EGP in {int(solution['iterations'])} "
        "iterations of one expectation, one analytic inverse marginal utility, and "
        "one univariate interpolation per income state. The two solutions agree to "
        f"{consumption_gap:.0e} on the asset range that holds the simulated mass — "
        "a discretization wedge, not a different economic object.\n\n"
        f"The economic content stays put: a Gini of {gini_wealth:.3f} on assets "
        "alone, average MPCs many times the perfect-foresight benchmark, and a "
        "non-trivial fraction of agents pinned at the borrowing limit. What EGP "
        "buys is the ability to use that household block as the inner step of a "
        "general-equilibrium fixed point or a structural estimation loop without "
        "the inner search becoming the binding cost — exactly the role it plays "
        "in the [Aiyagari](../../dynamic-programming/aiyagari/) and "
        "[Huggett](../huggett-incomplete-markets/) computations next door."
    )

    report.add_references(
        [
            "Carroll, C. D. (2006). The Method of Endogenous Gridpoints for Solving "
            "Dynamic Stochastic Optimization Problems. *Economics Letters*, 91(3), 312-320.",
            "Deaton, A. (1991). Saving and Liquidity Constraints. *Econometrica*, 59(5), 1221-1248.",
            "Carroll, C. D. (1997). Buffer-Stock Saving and the Life Cycle/Permanent Income "
            "Hypothesis. *Quarterly Journal of Economics*, 112(1), 1-55.",
            "Kaplan, G. and Violante, G. L. (2022). The Marginal Propensity to Consume in "
            "Heterogeneous Agent Models. *Annual Review of Economics*, 14, 747-775.",
        ]
    )

    report.write("README.md")
    print(
        f"\nGenerated: README.md + {len(report._figures)} figures + "
        f"{len(report._tables)} tables"
    )


if __name__ == "__main__":
    main()
