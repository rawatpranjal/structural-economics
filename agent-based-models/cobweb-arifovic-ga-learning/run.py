#!/usr/bin/env python3
"""Cobweb market with Arifovic genetic-algorithm learning.

A population of firms picks production quantities encoded as binary
chromosomes. Each generation is one market period: chromosomes decode to
quantities, the market clears, and selection / crossover / mutation /
election shape the next generation.

The classical cobweb is unstable when supply responds more strongly than
demand. Arifovic (1994) showed that the GA still converges to the
rational-expectations equilibrium (REE) in that regime, while naive
expectations diverge.

Reference: Arifovic, J. (1994), "Genetic algorithm learning and the cobweb
model," Journal of Economic Dynamics and Control 18(1): 3-28.
"""
from __future__ import annotations

import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


@dataclass
class MarketParams:
    """Linear demand a - b p and per-firm quadratic cost x q + (y/2) q^2."""
    a: float
    b: float
    x: float
    y: float
    n_firms: int

    @property
    def naive_slope(self) -> float:
        """|dp_t/dp_{t-1}| under naive expectations."""
        return self.n_firms / (self.b * self.y)

    @property
    def ree_price(self) -> float:
        return (self.a * self.y + self.n_firms * self.x) / (self.b * self.y + self.n_firms)

    @property
    def ree_quantity(self) -> float:
        return (self.ree_price - self.x) / self.y

    @property
    def is_stable(self) -> bool:
        return self.naive_slope < 1.0


@dataclass
class GAParams:
    chromosome_length: int = 8
    q_min: float = 0.0
    q_max: float = 2.0
    crossover_prob: float = 0.6
    mutation_prob: float = 0.02
    n_generations: int = 500
    use_election: bool = True


@dataclass
class GARun:
    params: MarketParams
    ga: GAParams
    prices: np.ndarray
    quantities: np.ndarray
    populations: list[np.ndarray] = field(default_factory=list)
    realized_profits: np.ndarray | None = None


def decode(chromosomes: np.ndarray, ga: GAParams) -> np.ndarray:
    """Map binary chromosomes to quantities in [q_min, q_max]."""
    weights = 1 << np.arange(ga.chromosome_length - 1, -1, -1)
    integers = chromosomes @ weights
    span = (1 << ga.chromosome_length) - 1
    return ga.q_min + (integers / span) * (ga.q_max - ga.q_min)


def market_price(total_q: float, params: MarketParams, demand_shift: float = 0.0) -> float:
    """Inverse demand at total quantity, with optional intercept shock."""
    return (params.a + demand_shift - total_q) / params.b


def firm_profit(q: np.ndarray, p: float, params: MarketParams) -> np.ndarray:
    return p * q - params.x * q - 0.5 * params.y * q * q


def naive_cobweb_path(
    params: MarketParams,
    p0: float,
    n_steps: int,
    demand_shifts: np.ndarray | None = None,
) -> np.ndarray:
    """Naive-expectations cobweb price path."""
    prices = np.empty(n_steps + 1)
    prices[0] = p0
    for t in range(n_steps):
        q_each = (prices[t] - params.x) / params.y
        total_q = max(params.n_firms * q_each, 0.0)
        shift = float(demand_shifts[t]) if demand_shifts is not None else 0.0
        prices[t + 1] = market_price(total_q, params, demand_shift=shift)
    return prices


def selection(profits: np.ndarray, rng: np.random.Generator, k: int = 3) -> np.ndarray:
    """Tournament selection: each parent is the winner of k random draws."""
    n = len(profits)
    candidates = rng.integers(0, n, size=(n, k))
    winner_in_tournament = np.argmax(profits[candidates], axis=1)
    return candidates[np.arange(n), winner_in_tournament]


def crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Single-point crossover."""
    length = parent_a.shape[0]
    point = int(rng.integers(1, length))
    child_a = np.concatenate([parent_a[:point], parent_b[point:]])
    child_b = np.concatenate([parent_b[:point], parent_a[point:]])
    return child_a, child_b


def mutate(chromosome: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    flips = rng.random(chromosome.shape[0]) < p
    return np.where(flips, 1 - chromosome, chromosome)


def reproduce(
    population: np.ndarray,
    profits: np.ndarray,
    realized_price: float,
    params: MarketParams,
    ga: GAParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the next-generation population.

    Selection -> crossover -> mutation -> Arifovic election operator.
    Election keeps a parent when its child's hypothetical profit at the
    just-realized price falls short of the parent's actual profit.
    """
    parent_indices = selection(profits, rng)
    parents = population[parent_indices]
    parent_profits = profits[parent_indices]
    new_pop = parents.copy()

    for i in range(0, params.n_firms, 2):
        if i + 1 >= params.n_firms:
            break
        a = parents[i]
        b = parents[i + 1]
        if rng.random() < ga.crossover_prob:
            ca, cb = crossover(a, b, rng)
        else:
            ca, cb = a.copy(), b.copy()
        ca = mutate(ca, ga.mutation_prob, rng)
        cb = mutate(cb, ga.mutation_prob, rng)

        if ga.use_election:
            children = np.stack([ca, cb])
            child_q = decode(children, ga)
            child_profit = firm_profit(child_q, realized_price, params)
            if child_profit[0] >= parent_profits[i]:
                new_pop[i] = ca
            if child_profit[1] >= parent_profits[i + 1]:
                new_pop[i + 1] = cb
        else:
            new_pop[i] = ca
            new_pop[i + 1] = cb
    return new_pop


def run_ga(
    params: MarketParams,
    ga: GAParams,
    seed: int = 0,
    demand_shifts: np.ndarray | None = None,
    keep_history: bool = True,
) -> GARun:
    rng = np.random.default_rng(seed)
    population = rng.integers(0, 2, size=(params.n_firms, ga.chromosome_length), dtype=np.int8)
    prices = np.empty(ga.n_generations)
    total_q = np.empty(ga.n_generations)
    populations: list[np.ndarray] = []
    last_profits = np.zeros(params.n_firms)

    for t in range(ga.n_generations):
        quantities = decode(population, ga)
        Q = float(quantities.sum())
        shift = float(demand_shifts[t]) if demand_shifts is not None else 0.0
        p = market_price(Q, params, demand_shift=shift)
        profits = firm_profit(quantities, p, params)
        prices[t] = p
        total_q[t] = Q
        last_profits = profits
        if keep_history:
            populations.append(decode(population, ga))
        population = reproduce(population, profits, p, params, ga, rng)

    return GARun(
        params=params,
        ga=ga,
        prices=prices,
        quantities=total_q,
        populations=populations,
        realized_profits=last_profits,
    )


def ols_with_hc0(
    quantities: np.ndarray,
    prices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """OLS of `quantities` on a constant and `prices`, with HC0 standard errors."""
    n = len(quantities)
    X = np.column_stack([np.ones(n), prices])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ quantities
    residuals = quantities - X @ beta
    Omega = (X.T * residuals**2) @ X
    cov = XtX_inv @ Omega @ XtX_inv
    return beta, np.sqrt(np.diag(cov))


def two_stage_least_squares(
    quantities: np.ndarray,
    prices: np.ndarray,
    instrument: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """2SLS for Q = c0 + c1 p + e using `instrument` for p.

    Returns (coefficients, HC0 standard errors).
    """
    n = len(quantities)
    Z = np.column_stack([np.ones(n), instrument])
    X = np.column_stack([np.ones(n), prices])
    p_hat = Z @ np.linalg.lstsq(Z, prices, rcond=None)[0]
    X_hat = np.column_stack([np.ones(n), p_hat])
    XtX_inv = np.linalg.inv(X_hat.T @ X_hat)
    beta = XtX_inv @ X_hat.T @ quantities
    residuals = quantities - X @ beta
    Omega = (X_hat.T * residuals**2) @ X_hat
    cov = XtX_inv @ Omega @ XtX_inv
    return beta, np.sqrt(np.diag(cov))


def plot_cobweb_panels(stable: MarketParams, unstable: MarketParams) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, params, title in [
        (axes[0], stable, f"Stable: $\\beta = {stable.naive_slope:.2f} < 1$"),
        (axes[1], unstable, f"Unstable: $\\beta = {unstable.naive_slope:.2f} > 1$"),
    ]:
        p_grid = np.linspace(0.5, 8.0, 200)
        q_demand = params.a - params.b * p_grid
        q_supply = params.n_firms * (p_grid - params.x) / params.y
        ax.plot(q_demand, p_grid, label="Demand $a-bp$")
        ax.plot(q_supply, p_grid, label="Supply $n(p-x)/y$")
        ax.axhline(params.ree_price, color="grey", linestyle=":", linewidth=1, label=r"REE $p^{\ast}$")

        p = 6.0 if params.is_stable else 4.6
        n_steps = 14
        for _ in range(n_steps):
            q_s = params.n_firms * (p - params.x) / params.y
            ax.plot([q_s, q_s], [p, market_price(q_s, params)], color="C3", linewidth=0.9)
            p_next = market_price(q_s, params)
            ax.plot([q_s, params.a - params.b * p_next], [p_next, p_next], color="C3", linewidth=0.9)
            p = p_next

        ax.set_xlim(0, max(q_demand.max(), 1.3 * params.n_firms * params.ree_quantity))
        ax.set_ylim(0, p_grid.max())
        ax.set_xlabel("Total quantity $Q$")
        ax.set_ylabel("Price $p$")
        ax.set_title(title)
        ax.legend(loc="upper right")
    fig.suptitle("Naive cobweb: stable spiral vs explosive divergence")
    fig.tight_layout()
    return fig


def plot_price_paths(
    stable_naive: np.ndarray,
    stable_ga: GARun,
    unstable_naive: np.ndarray,
    unstable_ga: GARun,
    show_steps: int = 80,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cases = [
        (axes[0], stable_naive, stable_ga, "Stable regime"),
        (axes[1], unstable_naive, unstable_ga, "Unstable regime"),
    ]
    for ax, naive_path, ga_run, title in cases:
        steps = np.arange(min(show_steps, len(ga_run.prices)))
        ax.plot(steps, naive_path[1 : len(steps) + 1], label="Naive cobweb", color="C1")
        ax.plot(steps, ga_run.prices[: len(steps)], label="GA learning", color="C0")
        ax.axhline(ga_run.params.ree_price, color="grey", linestyle=":", label=r"REE $p^{\ast}$")
        ax.set_xlabel("Period $t$")
        ax.set_ylabel("Market price $p_t$")
        ax.set_title(f"{title} ($\\beta={ga_run.params.naive_slope:.2f}$)")
        ax.legend(loc="upper right")
    fig.suptitle("GA learning settles near REE in both regimes; naive expectations explode in the unstable case")
    fig.tight_layout()
    return fig


def plot_chromosome_snapshots(run: GARun, snap_steps: list[int]) -> plt.Figure:
    fig, axes = plt.subplots(1, len(snap_steps), figsize=(3.2 * len(snap_steps), 3.6), sharey=True)
    bins = np.linspace(run.ga.q_min, run.ga.q_max, 25)
    for ax, t in zip(axes, snap_steps):
        ax.hist(run.populations[t], bins=bins, color="C0", edgecolor="white")
        ax.axvline(run.params.ree_quantity, color="C3", linestyle="--", label=r"REE $q^{\ast}$")
        ax.set_title(f"$t = {t}$")
        ax.set_xlabel("Quantity $q_i$")
    axes[0].set_ylabel("Number of firms")
    axes[0].legend(loc="upper right")
    fig.suptitle("Population quantity distribution concentrates around REE")
    fig.tight_layout()
    return fig


def population_frame(run: GARun, t: int, width: int = 720, height: int = 360) -> Image.Image:
    bins = np.linspace(run.ga.q_min, run.ga.q_max, 25)
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.hist(run.populations[t], bins=bins, color="C0", edgecolor="white")
    ax.axvline(run.params.ree_quantity, color="C3", linestyle="--", linewidth=2, label=r"REE $q^{\ast}$")
    ax.set_xlim(run.ga.q_min, run.ga.q_max)
    ax.set_ylim(0, run.params.n_firms)
    ax.set_xlabel("Quantity $q_i$")
    ax.set_ylabel("Firms")
    ax.set_title(rf"Generation $t = {t}$   |   $\bar p = {run.prices[t]:.2f}$, $p^{{\ast}} = {run.params.ree_price:.2f}$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_population_gif(run: GARun, path: str, n_frames: int = 32, duration_ms: int = 180) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indices = sorted(set(np.linspace(0, len(run.populations) - 1, n_frames, dtype=int)))
    frames = [population_frame(run, int(t)) for t in indices]
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def cobweb_frame(
    params: MarketParams,
    prices: np.ndarray,
    step: int,
    q_axis_lim: tuple[float, float],
    p_axis_lim: tuple[float, float],
) -> Image.Image:
    """Render one frame of the cobweb staircase up to `step` periods."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)

    p_grid = np.linspace(p_axis_lim[0], p_axis_lim[1], 200)
    q_demand = params.a - params.b * p_grid
    q_supply = params.n_firms * (p_grid - params.x) / params.y
    demand_visible = q_demand >= 0
    supply_visible = q_supply >= 0
    ax.plot(q_demand[demand_visible], p_grid[demand_visible], color="C0", linewidth=2, label="Demand $a - bp$")
    ax.plot(q_supply[supply_visible], p_grid[supply_visible], color="C1", linewidth=2, label="Supply $n(p-x)/y$")
    ax.scatter([params.n_firms * params.ree_quantity], [params.ree_price],
               color="black", zorder=5, s=40, label=r"REE $(Q^{\ast}, p^{\ast})$")

    for k in range(step):
        p_curr = float(prices[k])
        p_next = float(prices[k + 1])
        q_at_curr = max(params.n_firms * (p_curr - params.x) / params.y, 0.0)
        q_at_next = max(params.n_firms * (p_next - params.x) / params.y, 0.0)
        ax.plot([q_at_curr, q_at_curr], [p_curr, p_next], color="C3", linewidth=1.4, alpha=0.85)
        ax.plot([q_at_curr, q_at_next], [p_next, p_next], color="C3", linewidth=1.4, alpha=0.85)

    if step >= 1:
        p_curr = float(prices[step])
        q_curr = max(params.n_firms * (p_curr - params.x) / params.y, 0.0)
        ax.scatter([q_curr], [p_curr], color="C3", zorder=6, s=60)

    ax.set_xlim(*q_axis_lim)
    ax.set_ylim(*p_axis_lim)
    ax.set_xlabel(r"Total quantity $Q$")
    ax.set_ylabel(r"Price $p$")
    p_now = float(prices[min(step, len(prices) - 1)])
    ax.set_title(
        rf"Naive cobweb, $\beta = {params.naive_slope:.2f}$"
        "\n"
        rf"Step $t = {step}$   |   $p_t = {p_now:.2f}$, $p^{{\ast}} = {params.ree_price:.2f}$"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_cobweb_gif(
    params: MarketParams,
    prices: np.ndarray,
    path: str,
    duration_ms: int = 500,
    hold_last_ms: int = 1500,
) -> None:
    """Animate the cobweb staircase, one period per frame."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_steps = len(prices) - 1
    q_max = max(
        params.n_firms * (prices.max() - params.x) / params.y,
        params.a - params.b * max(prices.min(), 0.0),
    )
    q_pad = 0.05 * q_max
    q_axis_lim = (-q_pad, q_max + q_pad)

    p_max = float(max(prices.max(), params.ree_price)) * 1.15
    p_min = float(min(prices.min(), 0.0)) - 0.5
    p_axis_lim = (p_min, p_max)

    frames = [cobweb_frame(params, prices, step, q_axis_lim, p_axis_lim) for step in range(n_steps + 1)]
    durations = [duration_ms] * (len(frames) - 1) + [hold_last_ms]
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )


def plot_iv_recovery(
    quantities: np.ndarray,
    prices: np.ndarray,
    params: MarketParams,
    ols_beta: np.ndarray,
    iv_beta: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    p_grid = np.linspace(prices.min(), prices.max(), 100)
    axes[0].scatter(prices, quantities, s=10, alpha=0.45, color="C0", label="Cobweb market periods")
    axes[0].plot(p_grid, params.a - params.b * p_grid, color="black", label=f"True demand $a={params.a:.0f}, b={params.b:.0f}$")
    axes[0].plot(p_grid, ols_beta[0] + ols_beta[1] * p_grid, color="C1", linestyle="--", label=rf"Naive OLS fit $\hat b_{{\mathrm{{OLS}}}} = {-ols_beta[1]:.2f}$")
    axes[0].plot(p_grid, iv_beta[0] + iv_beta[1] * p_grid, color="C2", linestyle="--", label=rf"2SLS fit $\hat b_{{\mathrm{{IV}}}} = {-iv_beta[1]:.2f}$")
    axes[0].set_xlabel(r"Price $p_t$")
    axes[0].set_ylabel(r"Quantity $Q_t$")
    axes[0].set_title("Demand recovery: naive OLS biased by simultaneity")
    axes[0].legend(loc="upper right")

    labels = [r"Intercept $a$", r"Slope $b$"]
    truth = [params.a, params.b]
    ols_vals = [ols_beta[0], -ols_beta[1]]
    iv_vals = [iv_beta[0], -iv_beta[1]]
    pos = np.arange(len(labels))
    width = 0.27
    axes[1].bar(pos - width, truth, width, label="True", color="black")
    axes[1].bar(pos, ols_vals, width, label="Naive OLS", color="C1")
    axes[1].bar(pos + width, iv_vals, width, label="2SLS", color="C2")
    axes[1].set_xticks(pos)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Coefficient")
    axes[1].set_title("Lagged-price IV closes the bias")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    return fig


def main() -> None:
    setup_style()
    rng_master = np.random.default_rng(42)

    stable = MarketParams(a=60.0, b=30.0, x=1.0, y=2.0, n_firms=30)
    unstable = MarketParams(a=60.0, b=10.0, x=1.0, y=2.0, n_firms=30)
    ga_params = GAParams()

    naive_steps = 80
    p0 = unstable.ree_price + 1.5
    naive_unstable = naive_cobweb_path(unstable, p0=p0, n_steps=naive_steps)
    naive_stable = naive_cobweb_path(stable, p0=stable.ree_price + 1.5, n_steps=naive_steps)

    ga_run_unstable = run_ga(unstable, ga_params, seed=int(rng_master.integers(1 << 30)), keep_history=True)
    ga_run_stable = run_ga(stable, ga_params, seed=int(rng_master.integers(1 << 30)), keep_history=True)

    sigma = 0.5
    n_obs = 1500
    demand_shifts_iv = rng_master.normal(0.0, sigma, size=n_obs)
    iv_prices = naive_cobweb_path(
        stable, p0=stable.ree_price, n_steps=n_obs, demand_shifts=demand_shifts_iv,
    )
    iv_quantities = np.empty(n_obs + 1)
    for t in range(n_obs + 1):
        prev = iv_prices[t - 1] if t > 0 else iv_prices[0]
        iv_quantities[t] = stable.n_firms * (prev - stable.x) / stable.y

    burn = 50
    Q = iv_quantities[burn + 1 :]
    P = iv_prices[burn + 1 :]
    P_lag = iv_prices[burn : -1]
    ols_beta, ols_se = ols_with_hc0(Q, P)
    iv_beta, iv_se = two_stage_least_squares(Q, P, P_lag)

    iv_table = pd.DataFrame({
        "Parameter": ["Intercept $a$", "Slope $b$"],
        "True value": [stable.a, stable.b],
        "Naive OLS": [ols_beta[0], -ols_beta[1]],
        "OLS SE": [ols_se[0], ols_se[1]],
        "2SLS": [iv_beta[0], -iv_beta[1]],
        "2SLS SE": [iv_se[0], iv_se[1]],
    })

    regime_table = pd.DataFrame({
        "Regime": ["Stable", "Unstable"],
        "Demand slope $b$": [stable.b, unstable.b],
        "Supply slope $y$": [stable.y, unstable.y],
        "Firms $n$": [stable.n_firms, unstable.n_firms],
        "Naive slope $\\beta$": [stable.naive_slope, unstable.naive_slope],
        "REE price $p^{\\ast}$": [stable.ree_price, unstable.ree_price],
        "REE quantity $q^{\\ast}$": [stable.ree_quantity, unstable.ree_quantity],
        "Naive diverges": [not stable.is_stable, not unstable.is_stable],
        "GA mean $p_t$ (last 100)": [
            float(ga_run_stable.prices[-100:].mean()),
            float(ga_run_unstable.prices[-100:].mean()),
        ],
        "Distance to $p^{\\ast}$": [
            float(abs(ga_run_stable.prices[-100:].mean() - stable.ree_price)),
            float(abs(ga_run_unstable.prices[-100:].mean() - unstable.ree_price)),
        ],
    })

    cobweb_anim_unstable = naive_cobweb_path(unstable, p0=unstable.ree_price + 1.2, n_steps=14)
    save_cobweb_gif(unstable, cobweb_anim_unstable, "figures/cobweb-staircase.gif", duration_ms=600, hold_last_ms=1800)

    report = ModelReport(
        "Cobweb Markets and Arifovic Genetic-Algorithm Learning",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Strawberries take a season to grow. Farmers must decide how much to plant "
        "before the price is known, and many of them have to make that choice at the "
        "same time. The classical cobweb model puts a linear demand on top of a "
        "linear supply curve and assumes farmers expect the next price to equal the "
        "last one.\n\n"
        "When supply is more elastic than demand, this naive feedback loop is "
        "unstable. Each year overshoots the rational-expectations price by a larger "
        "amount and the market explodes. Arifovic (1994) asked whether a population "
        "of boundedly rational farmers, learning by genetic operators on binary "
        "production rules, could nevertheless settle on the rational price.\n\n"
        "This tutorial reproduces the headline result. We solve for the REE in "
        "closed form, simulate naive cobweb dynamics in a stable and an unstable "
        "regime, run Arifovic's GA on the same parameters, and finally take a "
        "noisy cobweb price series and recover the true demand curve via "
        "lagged-price IV."
    )

    report.add_equations(
        r"""
**Notation.**

| Symbol | What it means |
|---|---|
| $t$ | Period index. One season of strawberries. |
| $i$ | Firm index, running from $1$ to $n$. |
| $p_t$ | Market price in period $t$, the same for every buyer and every firm. |
| $q_{i,t}$ | Quantity firm $i$ chooses to plant for period $t$, decided before $p_t$ is known. |
| $Q_t$ | Total market quantity, $Q_t = \sum_{i=1}^{n} q_{i,t}$. |
| $\pi_{i,t}$ | Firm $i$'s realized profit in period $t$ once the market clears. |
| $a$ | Demand intercept. The price at which buyers stop buying ("choke price"). |
| $b$ | Demand slope. How many extra units buyers absorb per one-unit price cut. |
| $\varepsilon_t$ | Random demand shock in period $t$, mean zero, i.i.d. across periods. Stands in for weather, taste shifts, or news. |
| $x$ | Marginal-cost intercept. Cost of the very first unit a firm produces. |
| $y$ | Marginal-cost slope. How fast marginal cost rises as a firm produces more. |
| $n$ | Number of firms in the market. |
| $L$ | Length in bits of each firm's chromosome. |
| $N$ | GA population size. Equal to $n$ here, since one chromosome per firm. |
| $T$ | Number of GA generations the simulation runs. |
| $p_c$ | Per-pair crossover probability. |
| $p_m$ | Per-bit mutation probability. |

Per-firm cost is quadratic in own quantity: $C(q) = x q + \tfrac{y}{2} q^{2}$.

**Inverse demand with shock.** Market clearing gives

$$p_t = \frac{1}{b}\Big(\underbrace{a}_{\text{choke price}} + \underbrace{\varepsilon_t}_{\text{i.i.d. demand shock}} - \underbrace{Q_t}_{\text{aggregate quantity}}\Big).$$

**Firm supply.** Firm $i$ forms a price expectation $p_{i,t}^{e}$ before
producing, then sets $q_{i,t}$ to maximize expected profit. The first-order
condition gives the price-taking supply rule

$$q_{i,t} = \frac{p_{i,t}^{e} - x}{y}.$$

**Naive cobweb law of motion.** Plugging $p_{i,t}^{e} = p_{t-1}$ into the
supply rule for every firm and substituting into inverse demand gives a
one-step recursion in price:

$$p_t = \underbrace{\alpha}_{\text{intercept}} - \underbrace{\beta}_{\text{slope ratio}} \cdot p_{t-1}, \qquad \alpha = \frac{a y + n x}{b y}, \quad \beta = \frac{n}{b y}.$$

The fixed point of this recursion is the rational-expectations equilibrium

$$p^{\ast} = \frac{a y + n x}{b y + n}, \qquad q^{\ast} = \frac{p^{\ast} - x}{y}.$$

The naive cobweb converges to $p^{\ast}$ when $\beta < 1$ and explodes when
$\beta > 1$.

**Genetic-algorithm representation.** Each firm carries a length-$L$ binary
string $b_i \in \{0,1\}^{L}$ that decodes deterministically to a quantity in
the bracket $[q_{\min}, q_{\max}]$. The population size equals the number of
firms in the market, $N = n$, so each chromosome is one firm's production
plan in the current period.

**Fitness function.** The realized profit at the cleared price $p_t$ is

$$\pi_{i,t} = \underbrace{p_t \cdot q_{i,t}}_{\text{revenue}} - \underbrace{x \cdot q_{i,t}}_{\text{linear cost}} - \underbrace{\tfrac{y}{2} q_{i,t}^{2}}_{\text{convex cost}}.$$

**Abstract GA loop.** Let $\mathbf{B}_t = (b_{1,t}, \ldots, b_{n,t})$ be the
population of chromosomes at the start of period $t$. One generation
executes:

$$
\begin{aligned}
\text{(1) Decode} \quad & q_{i,t} = \mathrm{decode}(b_{i,t}). \\
\text{(2) Clear} \quad & Q_t = \sum_{i=1}^{n} q_{i,t}, \qquad p_t = \tfrac{a + \varepsilon_t - Q_t}{b}. \\
\text{(3) Score} \quad & \pi_{i,t} = p_t \cdot q_{i,t} - x \cdot q_{i,t} - \tfrac{y}{2} q_{i,t}^{2}. \\
\text{(4) Select} \quad & \text{tournament on } \{\pi_{i,t}\}_{i=1}^{n} \text{ produces parent indices.} \\
\text{(5) Recombine} \quad & \text{crossover with prob.\ } p_c, \text{ then bit-flip mutation with per-bit prob.\ } p_m. \\
\text{(6) Elect} \quad & \text{keep child } b_i' \text{ iff } \pi(\mathrm{decode}(b_i'), p_t) \geq \pi_{i,t}; \text{ else keep parent.} \\
\text{(7) Update} \quad & \mathbf{B}_{t+1} \leftarrow \text{surviving population.}
\end{aligned}
$$

The simulation runs for $T$ generations.
"""
    )

    report.add_model_setup(
        "Two regimes share the same demand intercept and per-firm cost, but differ "
        "in the demand slope $b$. The stable regime has $\\beta < 1$; the unstable "
        "regime has $\\beta > 1$ and naive expectations diverge.\n\n"
        f"| Object | Stable | Unstable | Role |\n"
        f"|---|---:|---:|---|\n"
        f"| Demand intercept $a$ | {stable.a:.0f} | {unstable.a:.0f} | Choke price |\n"
        f"| Demand slope $b$ | {stable.b:.0f} | {unstable.b:.0f} | Sensitivity of consumers to price |\n"
        f"| Cost intercept $x$ | {stable.x:.0f} | {unstable.x:.0f} | Marginal cost at zero output |\n"
        f"| Cost curvature $y$ | {stable.y:.0f} | {unstable.y:.0f} | Supply slope per firm |\n"
        f"| Number of firms $n$ | {stable.n_firms} | {unstable.n_firms} | Population size |\n"
        f"| Naive slope $\\beta$ | {stable.naive_slope:.2f} | {unstable.naive_slope:.2f} | Cobweb stability |\n"
        f"| REE price $p^{{*}}$ | {stable.ree_price:.2f} | {unstable.ree_price:.2f} | Fixed point |\n"
        f"| REE per-firm quantity $q^{{*}}$ | {stable.ree_quantity:.2f} | {unstable.ree_quantity:.2f} | Steady-state output |\n\n"
        "GA hyperparameters follow Arifovic's specification: chromosome length "
        f"$L = {ga_params.chromosome_length}$ giving "
        f"${1 << ga_params.chromosome_length}$ encoded quantity levels in "
        f"$[{ga_params.q_min:.0f}, {ga_params.q_max:.0f}]$, one chromosome per "
        f"firm so the population size equals $N = n = {stable.n_firms}$, "
        f"crossover probability $p_c = {ga_params.crossover_prob}$, mutation "
        f"probability per bit $p_m = {ga_params.mutation_prob}$, and "
        f"$T = {ga_params.n_generations}$ generations. The election operator is "
        "always on in the headline runs."
    )

    report.add_solution_method(
        "There is no closed-form solution for the GA dynamics. The model is "
        "solved by direct simulation. Each generation is one market period.\n\n"
        "```text\n"
        "Algorithm: Arifovic GA cobweb learning\n"
        "Input: market parameters (a, b, x, y, n), GA parameters (L, N, p_c, p_m, T)\n"
        "Output: price path p_1, ..., p_T and per-period population quantities\n\n"
        "1. Initialize N random binary chromosomes of length L.\n"
        "2. For t = 1 to T:\n"
        "   2a. Decode each chromosome into a quantity q_i in [q_min, q_max].\n"
        "   2b. Aggregate Q = sum_i q_i; clear the market at p_t = (a + e_t - Q) / b.\n"
        "   2c. Compute realized profit pi_i = p_t q_i - x q_i - (y/2) q_i^2.\n"
        "   2d. Roulette-select N parents in proportion to shifted profits.\n"
        "   2e. Pair parents and apply single-point crossover with prob p_c.\n"
        "   2f. Bit-flip mutate each child with prob p_m per bit.\n"
        "   2g. ELECTION: keep each child only if its profit at p_t exceeds its parent's profit.\n"
        "   2h. Replace the population with the survivors of step 2g.\n"
        "```\n\n"
        "The election step is the difference-maker. Without it, lucky offspring "
        "from a crossover that happened to land in a low-supply period propagate "
        "on inflated profits even though their implied quantity is far from "
        "the equilibrium. The election filter scores each child at the "
        "just-realized price and keeps it only if it beats its parent there.\n\n"
        "**Where this fits in evolutionary computation.** "
        "The genetic algorithm sits inside a wider family of evolutionary "
        "search methods. "
        "Holland (1975) introduced it as a population-based heuristic on "
        "fixed-length bit strings. "
        "Koza (1992) generalized to genetic programming, where the candidates "
        "are variable-length expressions or programs. "
        "Evolution strategies, CMA-ES, and neuroevolution swap the bit string "
        "for continuous parameters under Gaussian perturbations. "
        "Arifovic's contribution is the election operator. "
        "The operator filters each candidate child against a counterfactual "
        "profit at the just-realized market price. "
        "This filter is what stabilizes the otherwise unstable cobweb. "
        "It mirrors the policy-improvement step in Q-learning "
        "(see [`q-learning-growth`](../../dynamic-programming/q-learning-growth/)). "
        "In both, a learner accepts a candidate update only if it would have "
        "been an improvement under the most recently observed state."
    )

    report.add_results(
        "The cobweb diagram makes the stability story visible. In the stable "
        "regime, the staircase spirals inward to the supply-demand crossing. In "
        "the unstable regime, the same construction spirals outward and prices "
        "would explode if firms truly used last period's price as their forecast."
    )
    report.add_figure(
        "figures/cobweb-naive-vs-ree.png",
        "Naive cobweb staircase in the stable and unstable regimes",
        plot_cobweb_panels(stable, unstable),
    )

    report.add_results(
        "Watching the unstable cobweb draw itself one period at a time makes "
        "the divergence visceral. Each frame adds one supply-then-demand step "
        "to the spiral, and the price walks farther from $p^{\\ast}$ on every "
        "iteration.\n\n"
        '<img src="figures/cobweb-staircase.gif" alt="Animated naive cobweb staircase, unstable regime" width="70%">'
    )

    report.add_results(
        "Replacing naive expectations with the GA changes the picture. In the "
        "stable regime both rules behave similarly; the GA has a slightly noisier "
        "approach to REE because mutation never fully shuts off. In the unstable "
        "regime naive expectations diverge within a few periods while the GA "
        "settles into a tight band around the REE price."
    )
    report.add_figure(
        "figures/price-paths.png",
        "Naive vs GA price paths in both regimes",
        plot_price_paths(naive_stable, ga_run_stable, naive_unstable, ga_run_unstable),
    )

    report.add_results(
        "Looking inside the GA population shows what convergence means in this "
        "model. The initial chromosome distribution is uniform over the encoded "
        "quantity grid. Within a few dozen generations the bulk of firms are "
        "producing close to the REE quantity, and by generation 500 the "
        "population is concentrated in a narrow band around $q^{*}$."
    )
    report.add_figure(
        "figures/chromosome-snapshots.png",
        "Population quantity histograms at four generations (unstable regime)",
        plot_chromosome_snapshots(ga_run_unstable, snap_steps=[0, 25, 100, 499]),
    )

    report.add_results(
        "The estimation block uses a naive-cobweb price series with i.i.d. "
        "demand-intercept shocks $\\varepsilon_t$ as test data. The GA itself "
        "tracks REE so closely under the election operator that the resulting "
        "price barely moves. The naive cobweb provides the AR(1)-style "
        "persistence that makes the IV exercise interesting.\n\n"
        "**Structural demand equation.** The data are the realized pairs "
        "$\\{(p_t, Q_t)\\}_{t=1}^{T}$. The demand we want to recover is\n\n"
        "$$Q_t = a - b \\cdot p_t + \\varepsilon_t.$$\n\n"
        "**Why naive OLS is biased.** Market clearing forces the realized "
        "price to absorb the demand shock,\n\n"
        "$$p_t = \\frac{a + \\varepsilon_t - Q_t}{b} \\quad \\Longrightarrow \\quad \\mathrm{Cov}(p_t,\\, \\varepsilon_t) = \\frac{\\mathrm{Var}(\\varepsilon_t)}{b} > 0.$$\n\n"
        "An OLS regression of $Q_t$ on $p_t$ therefore underestimates the "
        "demand slope; the same simultaneity that drives prices in real "
        "markets drives them here.\n\n"
        "**Lagged-price IV.** The lagged price $p_{t-1}$ is correlated with "
        "$p_t$ through firms' naive supply rule but uncorrelated with the "
        "current shock under i.i.d. $\\varepsilon_t$,\n\n"
        "$$\\mathrm{Cov}(p_{t-1},\\, p_t) \\neq 0, \\qquad \\mathbb{E}[\\,p_{t-1} \\cdot \\varepsilon_t\\,] = 0.$$\n\n"
        "Two-stage least squares with $p_{t-1}$ as instrument is consistent. "
        "The first stage projects $p_t$ onto $(1, p_{t-1})$; the second stage "
        "regresses $Q_t$ on the fitted prices."
    )
    report.add_figure(
        "figures/iv-recovery.png",
        "Naive OLS vs lagged-price 2SLS demand-curve recovery",
        plot_iv_recovery(Q, P, stable, ols_beta, iv_beta),
    )

    report.add_table(
        "tables/parameter-grid.csv",
        "Regime grid and GA convergence summary",
        regime_table,
        "Naive cobweb stability is a knife-edge in $\\beta$. The GA tracks REE "
        "in both regimes; the absolute deviation of the last-100-period mean "
        "price from $p^{*}$ stays small even when the naive rule diverges.",
    )

    report.add_table(
        "tables/iv-estimates.csv",
        "Demand-curve recovery: true vs naive OLS vs 2SLS",
        iv_table,
        "Coefficients with HC0 standard errors. Naive OLS underestimates the "
        "demand slope because $\\varepsilon_t$ enters $p_t$ through market "
        "clearing. 2SLS with lagged price as instrument is consistent.",
    )

    report.add_takeaway(
        "Cobweb instability under naive expectations is a property of the "
        "aggregator, not of the agents. The same parameter grid that explodes "
        "under last-price forecasts converges under a population of binary "
        "learners, because the election operator filters out the lucky-price "
        "offspring whose decisions would not have been profitable in the current "
        "market.\n\n"
        "On the econometric side, the simulated cobweb price series sits in "
        "the same simultaneity geometry as a real market: demand shocks feed "
        "into the realized price through clearing, so a same-period "
        "regression cannot identify the demand curve. Lagged price is the "
        "textbook instrument and recovers the demand structure here, "
        "underlining that the identification logic depends on the timing of "
        "shocks more than on whether the supply side is strictly rational."
    )

    report.add_references([
        "[Arifovic, J. (1994). Genetic algorithm learning and the cobweb model. *Journal of Economic Dynamics and Control*, 18(1), 3-28.](https://doi.org/10.1016/0165-1889(94)90067-1)",
        "[Ezekiel, M. (1938). The cobweb theorem. *Quarterly Journal of Economics*, 52(2), 255-280.](https://doi.org/10.2307/1881734)",
        "[Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.]",
        "[Koza, J. R. (1992). *Genetic Programming: On the Programming of Computers by Means of Natural Selection*. MIT Press.]",
        "[Hansen, N. and Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. *Evolutionary Computation*, 9(2), 159-195.](https://doi.org/10.1162/106365601750190398)",
    ])

    report.write("README.md")


if __name__ == "__main__":
    main()
