#!/usr/bin/env python3
"""Allen-Arkolakis spatial equilibrium on a finite grid.

The source paper is Allen and Arkolakis (2014), Trade and the Topography of
the Spatial Economy. The paper works on a continuum of locations and then
derives a symmetric-cost Hammerstein equation. This tutorial keeps the same
economic blocks, gravity trade and labor mobility, but puts them on a finite
grid and solves the two equilibrium conditions directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import root

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


@dataclass(frozen=True)
class Geography:
    """Finite geography and exogenous fundamentals."""

    x: np.ndarray
    distance: np.ndarray
    trade_cost: np.ndarray
    abar: np.ndarray
    ubar: np.ndarray
    sigma: float
    kappa: float


@dataclass(frozen=True)
class Scenario:
    """Spillover parameters for a spatial-equilibrium scenario."""

    name: str
    alpha: float
    beta: float

    def gamma1(self, sigma: float) -> float:
        return 1.0 - (sigma - 1.0) * self.alpha - sigma * self.beta

    def gamma2(self, sigma: float) -> float:
        return 1.0 + sigma * self.alpha + (sigma - 1.0) * self.beta


@dataclass
class Equilibrium:
    """Solved equilibrium and diagnostics."""

    scenario: Scenario
    logw: np.ndarray
    labor: np.ndarray
    wages: np.ndarray
    productivity: np.ndarray
    amenity: np.ndarray
    price_index: np.ndarray
    trade_shares: np.ndarray
    income: np.ndarray
    sales: np.ndarray
    utility: np.ndarray
    market_access: np.ndarray
    success: bool
    message: str
    nfev: int
    max_trade_residual: float
    max_utility_residual: float


def make_geography(n_locations: int = 15, sigma: float = 5.0, kappa: float = 0.8) -> Geography:
    """Create a line economy with one central productivity advantage."""
    x = np.linspace(-1.0, 1.0, n_locations)
    distance = np.abs(x[:, None] - x[None, :])
    trade_cost = np.exp(kappa * distance)
    np.fill_diagonal(trade_cost, 1.0)

    central_bump = np.exp(-0.5 * (x / 0.35) ** 2)
    abar = np.exp(0.25 * central_bump)
    abar = abar / np.exp(np.mean(np.log(abar)))
    ubar = np.ones_like(x)
    return Geography(x, distance, trade_cost, abar, ubar, sigma, kappa)


def transport_cost_counterfactual(
    geo: Geography,
    kappa_multiplier: float = 0.75,
) -> Geography:
    """Lower iceberg trade costs while keeping fundamentals fixed."""
    kappa = geo.kappa * kappa_multiplier
    trade_cost = np.exp(kappa * geo.distance)
    np.fill_diagonal(trade_cost, 1.0)
    return Geography(
        x=geo.x.copy(),
        distance=geo.distance.copy(),
        trade_cost=trade_cost,
        abar=geo.abar.copy(),
        ubar=geo.ubar.copy(),
        sigma=geo.sigma,
        kappa=kappa,
    )


def labor_from_logits(logits: np.ndarray) -> np.ndarray:
    """Map R^(N-1) into the positive N-simplex."""
    padded = np.r_[logits, 0.0]
    padded = np.clip(padded - np.max(padded), -700.0, 700.0)
    weights = np.exp(padded)
    return weights / np.sum(weights)


def logits_from_labor(labor: np.ndarray) -> np.ndarray:
    """Use the last location as the labor-share reference category."""
    labor = np.asarray(labor, dtype=float)
    labor = np.maximum(labor, 1e-14)
    labor = labor / labor.sum()
    return np.log(labor[:-1] / labor[-1])


def equilibrium_objects(
    geo: Geography,
    logw: np.ndarray,
    labor: np.ndarray,
    scenario: Scenario,
) -> tuple[np.ndarray, ...]:
    """Compute prices, shares, income, and utility from wages and labor."""
    sigma = geo.sigma
    labor_safe = np.maximum(labor, 1e-300)
    wages = np.exp(logw)
    productivity = geo.abar * labor_safe ** scenario.alpha
    amenity = geo.ubar * labor_safe ** scenario.beta

    kernel = (
        geo.trade_cost ** (1.0 - sigma)
        * productivity[:, None] ** (sigma - 1.0)
        * wages[:, None] ** (1.0 - sigma)
    )
    price_power = kernel.sum(axis=0)
    trade_shares = kernel / price_power[None, :]
    price_index = price_power ** (1.0 / (1.0 - sigma))

    income = wages * labor
    sales = trade_shares @ income
    utility = wages * amenity / price_index
    market_access = (geo.trade_cost ** (1.0 - sigma)) @ income
    market_access = market_access / market_access.mean()
    return (
        wages,
        productivity,
        amenity,
        price_index,
        trade_shares,
        income,
        sales,
        utility,
        market_access,
    )


def residual_from_unknowns(z: np.ndarray, geo: Geography, scenario: Scenario) -> np.ndarray:
    """Root residuals: balanced trade, utility equalization, wage units."""
    n = len(geo.x)
    logw = z[:n]
    labor = labor_from_logits(z[n:])
    _, _, _, _, _, income, sales, utility, _ = equilibrium_objects(
        geo, logw, labor, scenario
    )
    trade_residual = np.log(income[:-1]) - np.log(sales[:-1])
    utility_residual = np.log(utility[1:]) - np.log(utility[0])
    wage_normalization = np.array([np.mean(logw)])
    return np.r_[trade_residual, utility_residual, wage_normalization]


def initial_unknowns(labor: np.ndarray) -> np.ndarray:
    """Build a root initial condition from a labor distribution."""
    return np.r_[np.zeros_like(labor), logits_from_labor(labor)]


def summarize_solution(
    geo: Geography,
    scenario: Scenario,
    sol,
) -> Equilibrium:
    """Convert a scipy root result into an equilibrium object."""
    n = len(geo.x)
    logw = sol.x[:n]
    logw = logw - np.mean(logw)
    labor = labor_from_logits(sol.x[n:])
    (
        wages,
        productivity,
        amenity,
        price_index,
        trade_shares,
        income,
        sales,
        utility,
        market_access,
    ) = equilibrium_objects(geo, logw, labor, scenario)
    max_trade_residual = float(np.max(np.abs(np.log(income) - np.log(sales))))
    max_utility_residual = float(
        np.max(np.abs(np.log(utility) - np.mean(np.log(utility))))
    )
    return Equilibrium(
        scenario=scenario,
        logw=logw,
        labor=labor,
        wages=wages,
        productivity=productivity,
        amenity=amenity,
        price_index=price_index,
        trade_shares=trade_shares,
        income=income,
        sales=sales,
        utility=utility,
        market_access=market_access,
        success=bool(sol.success),
        message=str(sol.message),
        nfev=int(getattr(sol, "nfev", -1)),
        max_trade_residual=max_trade_residual,
        max_utility_residual=max_utility_residual,
    )


def solve_equilibrium(
    geo: Geography,
    scenario: Scenario,
    starts: list[np.ndarray],
    z0: np.ndarray | None = None,
) -> tuple[Equilibrium, np.ndarray]:
    """Solve the full finite-grid equilibrium using scipy.optimize.root."""
    candidates: list[tuple[float, object]] = []
    initial_points = []
    if z0 is not None:
        initial_points.append(z0)
    initial_points.extend(initial_unknowns(start) for start in starts)

    for init in initial_points:
        sol = root(
            lambda z: residual_from_unknowns(z, geo, scenario),
            init,
            method="hybr",
            options={"xtol": 1e-11, "maxfev": 6000},
        )
        score = float(np.max(np.abs(residual_from_unknowns(sol.x, geo, scenario))))
        candidates.append((score, sol))

    successful = [(score, sol) for score, sol in candidates if sol.success]
    score, sol = min(successful or candidates, key=lambda item: item[0])
    return summarize_solution(geo, scenario, sol), sol.x


def solve_by_continuation(
    geo: Geography,
    scenarios: list[Scenario],
    starts: list[np.ndarray],
) -> tuple[dict[str, Equilibrium], dict[str, np.ndarray]]:
    """Walk across scenarios so the agglomeration case starts near a root."""
    out: dict[str, Equilibrium] = {}
    roots: dict[str, np.ndarray] = {}
    z0 = None
    for scenario in scenarios:
        eq, z0 = solve_equilibrium(geo, scenario, starts=starts, z0=z0)
        out[scenario.name] = eq
        roots[scenario.name] = z0
    return out, roots


def wage_residual_given_labor(
    logw: np.ndarray,
    geo: Geography,
    scenario: Scenario,
    labor: np.ndarray,
) -> np.ndarray:
    """Balanced-trade residual used inside the migration diagnostic."""
    _, _, _, _, _, income, sales, _, _ = equilibrium_objects(geo, logw, labor, scenario)
    return np.r_[np.log(income[:-1]) - np.log(sales[:-1]), np.mean(logw)]


def solve_wages_given_labor(
    geo: Geography,
    scenario: Scenario,
    labor: np.ndarray,
    logw_start: np.ndarray,
) -> np.ndarray:
    """Solve wages conditional on a provisional labor allocation."""
    sol = root(
        lambda logw: wage_residual_given_labor(logw, geo, scenario, labor),
        logw_start,
        method="hybr",
        options={"xtol": 1e-10, "maxfev": 2000},
    )
    logw = sol.x - np.mean(sol.x)
    return logw


def migration_path(
    geo: Geography,
    scenario: Scenario,
    start_labor: np.ndarray,
    iterations: int = 160,
    step_size: float = 0.8,
    damping: float = 0.25,
) -> dict[str, np.ndarray]:
    """A simple relocation dynamic used only as a teaching diagnostic."""
    labor = start_labor / start_labor.sum()
    logw = np.zeros_like(labor)
    labor_history = []
    gap_history = []

    for _ in range(iterations):
        logw = solve_wages_given_labor(geo, scenario, labor, logw)
        _, _, _, _, _, _, _, utility, _ = equilibrium_objects(
            geo, logw, labor, scenario
        )
        log_utility_gap = np.log(utility) - np.mean(np.log(utility))
        gap_history.append(float(np.max(np.abs(log_utility_gap))))
        labor_history.append(labor.copy())

        target = labor * np.exp(step_size * log_utility_gap)
        target = np.maximum(target, 1e-12)
        target = target / target.sum()
        labor = (1.0 - damping) * labor + damping * target
        labor = np.maximum(labor, 1e-12)
        labor = labor / labor.sum()

    return {
        "labor_history": np.asarray(labor_history),
        "gap_history": np.asarray(gap_history),
        "final_labor": labor,
    }


def make_starting_labor(geo: Geography) -> dict[str, np.ndarray]:
    """Named initial labor distributions."""
    x = geo.x
    starts = {
        "Uniform": np.ones_like(x),
        "Center": np.exp(-0.5 * (x / 0.25) ** 2) + 1e-2,
        "Left tilt": np.exp(-0.5 * ((x + 0.55) / 0.22) ** 2) + 1e-3,
        "Right tilt": np.exp(-0.5 * ((x - 0.55) / 0.22) ** 2) + 1e-3,
    }
    return {key: value / value.sum() for key, value in starts.items()}


def parameter_table(geo: Geography, scenarios: list[Scenario]) -> pd.DataFrame:
    """Build the parameter table for the report."""
    alpha_beta = ", ".join(
        f"{s.name}: alpha={s.alpha:.2f}, beta={s.beta:.2f}" for s in scenarios
    )
    gamma_values = ", ".join(
        f"{s.name}: gamma1={s.gamma1(geo.sigma):.2f}, gamma2={s.gamma2(geo.sigma):.2f}"
        for s in scenarios
    )
    return pd.DataFrame(
        [
            {
                "Symbol": "$N$",
                "Value": str(len(geo.x)),
                "Meaning": "Location count",
            },
            {
                "Symbol": "$x_i$",
                "Value": "15 equally spaced points in [-1, 1]",
                "Meaning": "Grid position",
            },
            {
                "Symbol": "$\\sigma$",
                "Value": f"{geo.sigma:.1f}",
                "Meaning": "Substitution elasticity",
            },
            {
                "Symbol": "$T_{ij}$",
                "Value": f"$\\exp({geo.kappa:.1f}\\lvert x_i-x_j\\rvert)$",
                "Meaning": "Iceberg trade cost",
            },
            {
                "Symbol": "$\\bar A_i$",
                "Value": "central log-productivity bump",
                "Meaning": "Productivity fundamental",
            },
            {
                "Symbol": "$\\bar u_i$",
                "Value": "1 for every location",
                "Meaning": "Amenity fundamental",
            },
            {
                "Symbol": "$\\alpha, \\beta$",
                "Value": alpha_beta,
                "Meaning": "Spillover parameters",
            },
            {
                "Symbol": "$\\gamma_1, \\gamma_2$",
                "Value": gamma_values,
                "Meaning": "Stability terms",
            },
            {
                "Symbol": "$\\sum_i L_i$",
                "Value": "1",
                "Meaning": "Labor normalization",
            },
            {
                "Symbol": "$\\frac{1}{N}\\sum_i \\log w_i$",
                "Value": "0",
                "Meaning": "Wage normalization",
            },
        ]
    )


def diagnostics_table(equilibria: dict[str, Equilibrium], geo: Geography) -> pd.DataFrame:
    """Summarize residuals and economic outcomes by scenario."""
    rows = []
    for eq in equilibria.values():
        rows.append(
            {
                "Scenario": eq.scenario.name,
                "alpha": f"{eq.scenario.alpha:.2f}",
                "beta": f"{eq.scenario.beta:.2f}",
                "gamma2/gamma1": f"{eq.scenario.gamma2(geo.sigma) / eq.scenario.gamma1(geo.sigma):.2f}",
                "Max trade residual": f"{eq.max_trade_residual:.2e}",
                "Max utility residual": f"{eq.max_utility_residual:.2e}",
                "Common utility": f"{np.exp(np.mean(np.log(eq.utility))):.4f}",
                "HHI": f"{np.sum(eq.labor ** 2):.3f}",
                "Largest share": f"{eq.labor.max():.3f}",
                "Solver": "converged" if eq.success else "check residual",
            }
        )
    return pd.DataFrame(rows)


def common_utility(eq: Equilibrium) -> float:
    """Compute the common real utility implied by a solved equilibrium."""
    return float(np.exp(np.mean(np.log(eq.utility))))


def counterfactual_table(
    baseline: dict[str, Equilibrium],
    policy: dict[str, Equilibrium],
    geo: Geography,
    policy_geo: Geography,
) -> pd.DataFrame:
    """Compare baseline geography with a lower trade-cost geography."""
    center = len(geo.x) // 2
    rows = []
    for name, base_eq in baseline.items():
        policy_eq = policy[name]
        base_welfare = common_utility(base_eq)
        policy_welfare = common_utility(policy_eq)
        base_hhi = float(np.sum(base_eq.labor ** 2))
        policy_hhi = float(np.sum(policy_eq.labor ** 2))
        base_center = float(base_eq.labor[center])
        policy_center = float(policy_eq.labor[center])
        rows.append(
            {
                "Scenario": name,
                "kappa change": f"{geo.kappa:.2f} to {policy_geo.kappa:.2f}",
                "Welfare change": f"{(policy_welfare / base_welfare - 1.0) * 100:.2f}%",
                "Policy welfare": f"{policy_welfare:.4f}",
                "HHI change": f"{policy_hhi - base_hhi:+.3f}",
                "Policy HHI": f"{policy_hhi:.3f}",
                "Center share change": f"{(policy_center - base_center) * 100:+.1f} pp",
                "Policy center share": f"{policy_center:.1%}",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    geo = make_geography()
    low_trade_geo = transport_cost_counterfactual(geo, kappa_multiplier=0.75)
    starts = make_starting_labor(geo)
    dispersion = Scenario("Dispersion dominant", alpha=0.03, beta=-0.12)
    bridge1 = Scenario("Bridge 1", alpha=0.05, beta=-0.08)
    bridge2 = Scenario("Bridge 2", alpha=0.07, beta=-0.05)
    agglomeration = Scenario("Agglomeration strong", alpha=0.12, beta=-0.02)

    solved, _ = solve_by_continuation(
        geo,
        [dispersion, bridge1, bridge2, agglomeration],
        starts=[starts["Uniform"], starts["Center"], starts["Left tilt"], starts["Right tilt"]],
    )
    equilibria = {
        dispersion.name: solved[dispersion.name],
        agglomeration.name: solved[agglomeration.name],
    }
    low_trade_solved, _ = solve_by_continuation(
        low_trade_geo,
        [dispersion, bridge1, bridge2, agglomeration],
        starts=[starts["Uniform"], starts["Center"], starts["Left tilt"], starts["Right tilt"]],
    )
    low_trade_equilibria = {
        dispersion.name: low_trade_solved[dispersion.name],
        agglomeration.name: low_trade_solved[agglomeration.name],
    }

    migration = {
        "dispersion_uniform": migration_path(geo, dispersion, starts["Uniform"]),
        "dispersion_left": migration_path(geo, dispersion, starts["Left tilt"]),
        "agglomeration_left": migration_path(geo, agglomeration, starts["Left tilt"]),
        "agglomeration_right": migration_path(geo, agglomeration, starts["Right tilt"]),
        "agglomeration_uniform": migration_path(geo, agglomeration, starts["Uniform"]),
    }

    setup_style()
    report = ModelReport(
        "Allen-Arkolakis Spatial Equilibrium on a Grid",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Spatial equilibrium models ask where workers live and what workers earn. "
        "They also ask how geography changes those outcomes. "
        "Allen and Arkolakis start from gravity trade, then add labor mobility.\n\n"
        "This tutorial puts 15 locations on a line. "
        "The center has a small productivity advantage. "
        "Distance raises shipping costs. "
        "Workers move until real utility is equalized. "
        "The tutorial compares a dispersed regime with an agglomerated regime. "
        "The comparison shows how spillovers change the spatial allocation."
    )

    report.add_equations(
        r"""
The model has a finite set of locations. I write this set as $i \in \lbrace 1,\ldots,N\rbrace$. I use $j$ when the same set is viewed as destinations.

At location $i$, $L_i$ is labor, $w_i$ is the wage, $A_i$ is productivity, and $u_i$ is the amenity. Trade costs are iceberg costs. Delivering one unit from $i$ to $j$ requires shipping $T_{ij}$ units from origin $i$.

The paper uses a continuum of locations. This tutorial uses a finite grid. That change turns integrals into sums over $i \in \lbrace 1,\ldots,N\rbrace$.

Productivity and amenities both start from fundamentals. They also respond to local labor:

$$
A_i = \bar A_i L_i^\alpha
$$

$$
u_i = \bar u_i L_i^\beta
$$

$\alpha$ is the productivity spillover. It is positive here. A larger $L_i$ raises $A_i$.

$\beta$ is the congestion parameter. It is negative here. A larger $L_i$ lowers $u_i$.

Consumers buy varieties from all origins. For destination $j$, the CES price index is

$$
P_j^{1-\sigma}
= \sum_i T_{ij}^{1-\sigma} A_i^{\sigma-1} w_i^{1-\sigma}.
$$

The spending share $\pi_{ij}$ is destination $j$'s spending on goods from origin $i$:

$$
\pi_{ij} =
\frac{T_{ij}^{1-\sigma} A_i^{\sigma-1} w_i^{1-\sigma}}
{\sum_k T_{kj}^{1-\sigma} A_k^{\sigma-1} w_k^{1-\sigma}}.
$$

Balanced trade says income at origin $i$ equals its sales across destinations:

$$
w_i L_i =
\sum_j \pi_{ij} w_j L_j.
$$

Mobility says workers are indifferent across inhabited locations:

$$
\frac{w_i u_i}{P_i} = V.
$$

Here $V$ is the common real utility level.

$$
\sum_i L_i = 1
$$

$$
N^{-1}\sum_i \log w_i = 0.
$$

The first normalization sets total labor to one. The second normalization sets the average log wage to zero.

The tutorial solves balanced trade and mobility directly. This matches the finite-location version of equations (11) and (12) in Allen and Arkolakis. It does not use the later Hammerstein reduction, which uses symmetry to eliminate wages.
"""
    )

    report.add_model_setup(
        "| Symbol | Value | Role |\n"
        "|--------|-------|------|\n"
        f"| $N$ | {len(geo.x)} | Location count |\n"
        f"| $x_i$ | equally spaced in $[-1,1]$ | Grid position |\n"
        f"| $\\sigma$ | {geo.sigma:.1f} | Substitution elasticity |\n"
        f"| $T_{{ij}}$ | $\\exp({geo.kappa:.1f}\\lvert x_i-x_j\\rvert)$ | Iceberg trade cost |\n"
        "| $\\bar A_i$ | central bump | Productivity fundamental |\n"
        "| $\\bar u_i$ | 1 | Amenity fundamental |\n"
        f"| $\\alpha,\\beta$ baseline | {dispersion.alpha:.2f}, {dispersion.beta:.2f} | Dispersion regime |\n"
        f"| $\\alpha,\\beta$ strong agglomeration | {agglomeration.alpha:.2f}, {agglomeration.beta:.2f} | Agglomeration regime |\n"
        "| Total labor | 1 | Labor normalization |\n"
        "| Wage normalization | geometric mean wage one | Wage units |"
    )

    report.add_solution_method(
        "The solver works with log wages and labor logits. "
        "A softmax maps logits into labor shares. "
        "This keeps wages positive, labor positive, and total labor equal to one.\n\n"
        "```text\n"
        "Algorithm: finite-grid spatial equilibrium\n"
        "Input : T_ij, Abar_i, ubar_i, alpha, beta\n"
        "Output: w_i, L_i\n"
        "  choose log wage unknowns omega_i\n"
        "  choose labor logits z_i\n"
        "  map z into L by softmax\n"
        "  compute A_i = Abar_i L_i^alpha\n"
        "  compute u_i = ubar_i L_i^beta\n"
        "  compute CES price indexes P_j\n"
        "  compute trade shares pi_ij\n"
        "  residual 1: log(w_i L_i) - log(sum_j pi_ij w_j L_j)\n"
        "  residual 2: log(w_i u_i / P_i) - log(w_1 u_1 / P_1)\n"
        "  residual 3: mean_i log w_i = 0\n"
        "  solve residuals = 0\n"
        "```\n\n"
        "The high-agglomeration case uses continuation. "
        "Continuation starts at the dispersion root and changes spillover parameters in two steps. "
        "This helps the root search. "
        "It is not an economic assumption."
    )

    # Figure 1: geography and fundamentals.
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.plot(geo.x, geo.abar, marker="o", color="tab:blue", label=r"$\bar A_i$")
    ax1.plot(geo.x, geo.ubar, marker="s", color="tab:green", label=r"$\bar u_i$")
    ax1.set_xlabel("Location on the line")
    ax1.set_ylabel("Fundamental, normalized")
    ax1.set_title("Fundamentals")
    ax1.legend()

    center = len(geo.x) // 2
    ax2.plot(
        geo.x,
        geo.trade_cost[center, :],
        marker="o",
        color="tab:red",
        label=r"$T_{\mathrm{center},j}$",
    )
    ax2.plot(
        geo.x,
        geo.trade_cost[center, :] ** (1.0 - geo.sigma),
        marker="s",
        color="tab:purple",
        label=r"$T_{\mathrm{center},j}^{1-\sigma}$",
    )
    ax2.set_xlabel("Destination")
    ax2.set_ylabel("Trade-cost term")
    ax2.set_title("Shipping from the center")
    ax2.legend()
    fig1.tight_layout()

    report.add_results(
        "The grid is deliberately small. "
        "The center has the only built-in location advantage. "
        "Trade costs rise with distance. "
        "Distant destinations therefore receive lower weight in price indexes and sales."
    )
    report.add_figure(
        "figures/fundamentals.png",
        "Fundamentals plus trade costs",
        fig1,
    )

    # Figure 2: equilibrium wages and labor.
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {
        dispersion.name: "tab:blue",
        agglomeration.name: "tab:orange",
    }
    for name, eq in equilibria.items():
        ax3.plot(geo.x, eq.wages, marker="o", label=name, color=colors[name])
        ax4.plot(geo.x, eq.labor, marker="o", label=name, color=colors[name])
    ax3.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax3.set_xlabel("Location on the line")
    ax3.set_ylabel("Wage, geometric mean one")
    ax3.set_title("Equilibrium wages")
    ax3.legend()
    ax4.set_xlabel("Location on the line")
    ax4.set_ylabel("Labor share")
    ax4.set_title("Equilibrium population")
    ax4.legend()
    fig2.tight_layout()

    disp_eq = equilibria[dispersion.name]
    agg_eq = equilibria[agglomeration.name]
    report.add_results(
        f"The dispersion-dominant center share is {disp_eq.labor.max():.1%}. "
        f"The strong-agglomeration largest share is {agg_eq.labor.max():.1%}. "
        "Nominal wages, price indexes, and amenities differ across space. "
        "Real utility is still equalized."
    )
    report.add_figure(
        "figures/equilibrium-wages-population.png",
        "Wages plus population shares",
        fig2,
    )

    # Figure 3: trade-cost heatmap and market access.
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 4.8))
    im = ax5.imshow(
        np.log(geo.trade_cost),
        origin="lower",
        cmap="magma",
        extent=[geo.x.min(), geo.x.max(), geo.x.min(), geo.x.max()],
        aspect="auto",
    )
    ax5.set_xlabel("Destination")
    ax5.set_ylabel("Origin")
    ax5.set_title(r"Log trade cost $\log T_{ij}$")
    fig3.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
    for name, eq in equilibria.items():
        ax6.plot(geo.x, eq.market_access, marker="o", label=name, color=colors[name])
    ax6.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax6.set_xlabel("Origin")
    ax6.set_ylabel("Market access, mean one")
    ax6.set_title("Demand access")
    ax6.legend()
    fig3.tight_layout()

    report.add_results(
        "The heatmap fixes geography. "
        "Purchasing power moves across locations. "
        "Central concentration raises demand access for nearby origins. "
        "Those origins serve more workers at lower shipping costs."
    )
    report.add_figure(
        "figures/access-surface.png",
        "Trade costs plus market access",
        fig3,
    )

    # Figure 4: convergence and path dependence diagnostic.
    fig4, (ax7, ax8) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax7.semilogy(
        migration["dispersion_uniform"]["gap_history"],
        color="tab:blue",
        label="Dispersion, uniform start",
    )
    ax7.semilogy(
        migration["dispersion_left"]["gap_history"],
        color="tab:cyan",
        linestyle="--",
        label="Dispersion, tilted start",
    )
    ax7.semilogy(
        migration["agglomeration_left"]["gap_history"],
        color="tab:orange",
        label="Agglomeration, left start",
    )
    ax7.semilogy(
        migration["agglomeration_right"]["gap_history"],
        color="tab:red",
        linestyle="--",
        label="Agglomeration, right start",
    )
    ax7.set_xlabel("Relocation iteration")
    ax7.set_ylabel("Max log utility gap")
    ax7.set_title("Teaching fixed point")
    ax7.legend(fontsize=8)

    ax8.plot(
        geo.x,
        migration["agglomeration_left"]["final_labor"],
        marker="o",
        color="tab:orange",
        label="Left start",
    )
    ax8.plot(
        geo.x,
        migration["agglomeration_right"]["final_labor"],
        marker="s",
        color="tab:red",
        label="Right start",
    )
    ax8.plot(
        geo.x,
        migration["agglomeration_uniform"]["final_labor"],
        marker="^",
        color="tab:gray",
        label="Uniform start",
    )
    ax8.set_xlabel("Location on the line")
    ax8.set_ylabel("Labor share after diagnostic")
    ax8.set_title("Agglomeration paths")
    ax8.legend(fontsize=8)
    fig4.tight_layout()

    left_final = migration["agglomeration_left"]["final_labor"]
    right_final = migration["agglomeration_right"]["final_labor"]
    path_gap = float(np.max(np.abs(left_final - right_final)))
    report.add_results(
        "The relocation iteration is diagnostic only. "
        "It solves wages for a provisional population and computes real utility gaps. "
        "It then shifts workers toward high-utility locations.\n\n"
        "Under dispersion dominance, both starts converge quickly. "
        "Under strong agglomeration, left and right starts remain different. "
        f"Their final labor profiles differ by as much as {path_gap:.3f}. "
        "This illustrates non-uniqueness. "
        "Stronger spillovers can remove global uniqueness."
    )
    report.add_figure(
        "figures/convergence-path-dependence.png",
        "Relocation diagnostic",
        fig4,
    )

    params = parameter_table(geo, [dispersion, agglomeration])
    diagnostics = diagnostics_table(equilibria, geo)
    counterfactuals = counterfactual_table(
        equilibria,
        low_trade_equilibria,
        geo,
        low_trade_geo,
    )
    report.add_results(
        "The parameter table records the normalizations and spillover regimes. "
        "The diagnostic table reports the two residual blocks and concentration."
    )
    report.add_table(
        "tables/parameters.csv",
        "Parameter table",
        params,
    )
    report.add_table(
        "tables/scenario-diagnostics.csv",
        "Equilibrium diagnostics by scenario",
        diagnostics,
    )
    report.add_results(
        f"The policy experiment lowers the trade-cost slope from {geo.kappa:.2f} to "
        f"{low_trade_geo.kappa:.2f}. "
        "This represents lower transport costs. "
        "Lower transport costs raise welfare in both regimes through better access. "
        "Population also moves, and the direction depends on the spillover regime."
    )
    report.add_table(
        "tables/trade-cost-counterfactual.csv",
        "Lower trade-cost counterfactual",
        counterfactuals,
    )

    report.add_takeaway(
        "For policy, the model tracks welfare, concentration, and geographic redistribution. "
        "Lower trade costs improve access and raise real utility. "
        "They can also move activity across space.\n\n"
        "Agglomeration can raise productivity, but it can also create congestion and concentration risk. "
        "Strong dispersion makes outcomes more predictable. "
        "Strong agglomeration makes history and initial conditions more important.\n\n"
        "Transport policy should be judged by welfare, concentration, and redistribution together. "
        "A higher common utility number is not the whole policy answer. "
        "It does not say which locations gain population. "
        "It does not measure concentration risk."
    )

    report.add_references(
        [
            "Allen, T. and Arkolakis, C. (2014). *Trade and the Topography of the Spatial Economy*. Quarterly Journal of Economics 129(3), 1085-1140. https://doi.org/10.1093/qje/qju016.",
            "Allen, T. and Arkolakis, C. (2013). *Trade and the Topography of the Spatial Economy*. NBER Working Paper 19181. https://www.nber.org/papers/w19181.",
            "Redding, S. J. and Rossi-Hansberg, E. (2017). *Quantitative Spatial Economics*. Annual Review of Economics 9, 21-58.",
        ]
    )

    report.write("README.md")
    print(
        f"Generated: README.md + {len(report._figures)} figures + "
        f"{len(report._tables)} tables"
    )


if __name__ == "__main__":
    main()
