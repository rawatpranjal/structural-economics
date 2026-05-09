#!/usr/bin/env python3
"""Small Q-learning duopoly pricing experiment.

This is a compact teaching version of the algorithmic-collusion environment
studied by Calvano, Calzolari, Denicolo, and Pastorello. Two firms repeatedly
choose prices for differentiated products. Demand is logit with an outside
option. Each firm learns a tabular pricing policy from realized profit feedback.
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
class Params:
    """Primitive and learning parameters for the duopoly pricing game."""

    a: float = 2.0
    a0: float = 0.0
    mu: float = 0.25
    c: float = 1.0
    k: int = 11
    steps: int = 80_000
    delta: float = 0.95
    eta: float = 0.12
    eps_min: float = 0.02
    eps_tau: float = 12_000.0
    eval_periods: int = 120
    eval_burn: int = 60
    deviation_horizon: int = 25


@dataclass(frozen=True)
class Benchmarks:
    """Static price benchmarks and the discrete price grid."""

    p_bertrand: np.ndarray
    p_monopoly: np.ndarray
    grid: np.ndarray
    bertrand_index: int
    low_deviation_index: int
    profit_table: np.ndarray

    @property
    def bertrand_price(self) -> float:
        return float(np.mean(self.p_bertrand))

    @property
    def monopoly_price(self) -> float:
        return float(np.mean(self.p_monopoly))


@dataclass
class SeedRun:
    """Learned policy diagnostics for one random seed."""

    seed: int
    q: np.ndarray
    greedy_actions: np.ndarray
    greedy_prices: np.ndarray
    greedy_profits: np.ndarray
    learned_price: float
    learned_profit: float
    collusion_index: float
    deviation_prices: np.ndarray
    pre_deviation_price: float
    min_post_deviation_price: float
    recovery_horizon: int | None


def demand(prices: np.ndarray, params: Params) -> np.ndarray:
    """Logit shares for the two inside goods."""
    prices = np.asarray(prices, dtype=float)
    quality = np.exp(np.clip((params.a - prices) / params.mu, -700.0, 700.0))
    outside = np.exp(params.a0 / params.mu)
    return quality / (quality.sum() + outside)


def profits(prices: np.ndarray, params: Params) -> np.ndarray:
    """Per-firm static profit at a two-price vector."""
    prices = np.asarray(prices, dtype=float)
    return (prices - params.c) * demand(prices, params)


def bertrand_foc(prices: np.ndarray, params: Params) -> np.ndarray:
    """Static differentiated-products Bertrand first-order conditions."""
    prices = np.asarray(prices, dtype=float)
    shares = demand(prices, params)
    return 1.0 - (prices - params.c) * (1.0 - shares) / params.mu


def monopoly_foc(prices: np.ndarray, params: Params) -> np.ndarray:
    """Joint-monopoly first-order conditions for the two-product firm."""
    prices = np.asarray(prices, dtype=float)
    shares = demand(prices, params)
    return np.array([
        1.0
        - (prices[0] - params.c) * (1.0 - shares[0]) / params.mu
        + (prices[1] - params.c) * shares[1] / params.mu,
        1.0
        - (prices[1] - params.c) * (1.0 - shares[1]) / params.mu
        + (prices[0] - params.c) * shares[0] / params.mu,
    ])


def solve_benchmarks(params: Params) -> Benchmarks:
    """Solve static benchmarks and build the padded action grid."""
    p0 = np.full(2, params.c + 1.0)
    bertrand = root(lambda p: bertrand_foc(p, params), p0)
    monopoly = root(lambda p: monopoly_foc(p, params), p0)
    if not bertrand.success or not monopoly.success:
        raise RuntimeError("Static price benchmark solve failed")

    p_bertrand = np.asarray(bertrand.x, dtype=float)
    p_monopoly = np.asarray(monopoly.x, dtype=float)
    core_grid = np.linspace(p_bertrand.min(), p_monopoly.max(), params.k - 2)
    step = core_grid[1] - core_grid[0]
    grid = np.linspace(core_grid[0] - step, core_grid[-1] + step, params.k)
    bertrand_index = int(np.argmin(np.abs(grid - p_bertrand.mean())))
    low_deviation_index = max(0, bertrand_index - 1)

    profit_table = np.zeros((params.k, params.k, 2))
    for i, p_i in enumerate(grid):
        for j, p_j in enumerate(grid):
            profit_table[i, j] = profits(np.array([p_i, p_j]), params)

    return Benchmarks(
        p_bertrand=p_bertrand,
        p_monopoly=p_monopoly,
        grid=grid,
        bertrand_index=bertrand_index,
        low_deviation_index=low_deviation_index,
        profit_table=profit_table,
    )


def initialize_q(bench: Benchmarks, params: Params) -> np.ndarray:
    """Optimistic continuation values based on average current profits."""
    q = np.zeros((2, params.k, params.k, params.k))
    own_profit_by_action = [
        bench.profit_table[:, :, 0].mean(axis=1),
        bench.profit_table[:, :, 1].mean(axis=0),
    ]
    for firm, expected_profit in enumerate(own_profit_by_action):
        q[firm] = np.tile(expected_profit.reshape(1, 1, params.k), (params.k, params.k, 1))
    return q / (1.0 - params.delta)


def greedy_action(q: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Greedy action pair at a previous-price state."""
    return np.array([
        int(np.argmax(q[0, state[0], state[1]])),
        int(np.argmax(q[1, state[0], state[1]])),
    ])


def train_q_learning(seed: int, bench: Benchmarks, params: Params) -> np.ndarray:
    """Train independent tabular Q-learning policies."""
    rng = np.random.default_rng(seed)
    q = initialize_q(bench, params)
    state = np.array([bench.bertrand_index, bench.bertrand_index], dtype=int)

    for t in range(params.steps):
        epsilon = params.eps_min + (1.0 - params.eps_min) * np.exp(-t / params.eps_tau)
        action = np.empty(2, dtype=int)
        for firm in range(2):
            if rng.random() < epsilon:
                action[firm] = int(rng.integers(params.k))
            else:
                values = q[firm, state[0], state[1]]
                best = np.flatnonzero(np.isclose(values, values.max()))
                action[firm] = int(rng.choice(best))

        reward = bench.profit_table[action[0], action[1]]
        next_state = action.copy()
        for firm in range(2):
            old_value = q[firm, state[0], state[1], action[firm]]
            continuation = q[firm, next_state[0], next_state[1]].max()
            target = reward[firm] + params.delta * continuation
            q[firm, state[0], state[1], action[firm]] = (
                (1.0 - params.eta) * old_value + params.eta * target
            )
        state = next_state

    return q


def greedy_rollout(q: np.ndarray, bench: Benchmarks, periods: int, start: np.ndarray | None = None) -> np.ndarray:
    """Roll out the learned greedy policy."""
    if start is None:
        state = np.array([bench.bertrand_index, bench.bertrand_index], dtype=int)
    else:
        state = np.asarray(start, dtype=int).copy()
    actions = np.empty((periods, 2), dtype=int)
    for t in range(periods):
        action = greedy_action(q, state)
        actions[t] = action
        state = action
    return actions


def deviation_rollout(q: np.ndarray, bench: Benchmarks, params: Params) -> tuple[np.ndarray, float, float, int | None]:
    """Force firm 1 to undercut once, then return to greedy play."""
    pre_actions = greedy_rollout(q, bench, 40)
    pre_state = pre_actions[-1]
    pre_price = float(bench.grid[pre_actions[-20:]].mean())

    first_action = np.array([
        bench.low_deviation_index,
        int(np.argmax(q[1, pre_state[0], pre_state[1]])),
    ])
    actions = [first_action]
    state = first_action.copy()
    for _ in range(params.deviation_horizon):
        action = greedy_action(q, state)
        actions.append(action)
        state = action

    deviation_prices = bench.grid[np.asarray(actions)]
    average_prices = deviation_prices.mean(axis=1)
    min_post = float(average_prices[1:].min())
    recovered = np.flatnonzero(average_prices[1:] >= 0.95 * pre_price)
    recovery_horizon = int(recovered[0] + 1) if recovered.size else None
    return deviation_prices, pre_price, min_post, recovery_horizon


def summarize_seed(seed: int, bench: Benchmarks, params: Params) -> SeedRun:
    """Train and evaluate one seed."""
    q = train_q_learning(seed, bench, params)
    actions = greedy_rollout(q, bench, params.eval_periods)
    prices = bench.grid[actions]
    period_profits = np.array([bench.profit_table[a[0], a[1]] for a in actions])
    eval_prices = prices[-params.eval_burn:]
    eval_profits = period_profits[-params.eval_burn:]

    learned_price = float(eval_prices.mean())
    learned_profit = float(eval_profits.mean())
    collusion_index = (learned_price - bench.bertrand_price) / (
        bench.monopoly_price - bench.bertrand_price
    )
    deviation_prices, pre_price, min_post, horizon = deviation_rollout(q, bench, params)
    return SeedRun(
        seed=seed,
        q=q,
        greedy_actions=actions,
        greedy_prices=prices,
        greedy_profits=period_profits,
        learned_price=learned_price,
        learned_profit=learned_profit,
        collusion_index=float(collusion_index),
        deviation_prices=deviation_prices,
        pre_deviation_price=pre_price,
        min_post_deviation_price=min_post,
        recovery_horizon=horizon,
    )


def plot_price_paths(runs: list[SeedRun], bench: Benchmarks, params: Params) -> plt.Figure:
    """Greedy learned price paths for representative seeds."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    time = np.arange(params.eval_periods)
    for run, color in zip(runs, ["C0", "C1", "C2", "C3", "C4"]):
        avg_price = run.greedy_prices.mean(axis=1)
        ax.plot(time, avg_price, color=color, linewidth=1.8, label=f"seed {run.seed}")
    ax.axhline(bench.bertrand_price, color="black", linestyle=":", linewidth=1.2, label="Bertrand-Nash")
    ax.axhline(bench.monopoly_price, color="black", linestyle="--", linewidth=1.2, label="Joint monopoly")
    ax.set_xlabel("Greedy play period after training")
    ax.set_ylabel("Average price")
    ax.set_title("Learned greedy prices are above the static Bertrand benchmark")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    return fig


def plot_deviation_response(runs: list[SeedRun], bench: Benchmarks) -> plt.Figure:
    """Forced one-firm deviation and subsequent greedy response."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    horizon = runs[0].deviation_prices.shape[0]
    time = np.arange(horizon)

    for run in runs:
        axes[0].plot(time, run.deviation_prices[:, 0], color="C3", alpha=0.22, linewidth=1.2)
        axes[1].plot(time, run.deviation_prices[:, 1], color="C0", alpha=0.22, linewidth=1.2)

    firm_1 = np.array([run.deviation_prices[:, 0] for run in runs])
    firm_2 = np.array([run.deviation_prices[:, 1] for run in runs])
    axes[0].plot(time, firm_1.mean(axis=0), color="C3", linewidth=2.5, label="mean across seeds")
    axes[1].plot(time, firm_2.mean(axis=0), color="C0", linewidth=2.5, label="mean across seeds")

    for ax, title in zip(axes, ["Deviating firm", "Rival firm"]):
        ax.axhline(bench.bertrand_price, color="black", linestyle=":", linewidth=1.1, label="Bertrand-Nash")
        ax.axhline(bench.monopoly_price, color="black", linestyle="--", linewidth=1.1, label="Joint monopoly")
        ax.set_xlabel("Periods after forced deviation")
        ax.set_title(title)
        ax.legend(loc="lower right")
    axes[0].set_ylabel("Price")
    fig.suptitle("Forced low-price deviation: responses are visible but short and seed-dependent")
    fig.tight_layout()
    return fig


def plot_learning_diagnostics(runs: list[SeedRun], bench: Benchmarks) -> plt.Figure:
    """Seed-level learned price and profit diagnostics."""
    seeds = [str(run.seed) for run in runs]
    learned_prices = np.array([run.learned_price for run in runs])
    collusion = np.array([run.collusion_index for run in runs])
    profit_ratio = np.array([run.learned_profit for run in runs])
    competitive_profit = bench.profit_table[bench.bertrand_index, bench.bertrand_index].mean()
    monopoly_index = int(np.argmin(np.abs(bench.grid - bench.monopoly_price)))
    monopoly_profit = bench.profit_table[monopoly_index, monopoly_index].mean()
    profit_ratio = (profit_ratio - competitive_profit) / (monopoly_profit - competitive_profit)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(runs))
    axes[0].bar(x, learned_prices, color="C0", alpha=0.85)
    axes[0].axhline(bench.bertrand_price, color="black", linestyle=":", linewidth=1.1, label="Bertrand-Nash")
    axes[0].axhline(bench.monopoly_price, color="black", linestyle="--", linewidth=1.1, label="Joint monopoly")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(seeds)
    axes[0].set_xlabel("Training seed")
    axes[0].set_ylabel("Average learned price")
    axes[0].set_title("Learned prices by seed")
    axes[0].legend(loc="upper left")

    width = 0.36
    axes[1].bar(x - width / 2, collusion, width, color="C2", label="Price collusion index")
    axes[1].bar(x + width / 2, profit_ratio, width, color="C4", label="Profit ratio")
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(seeds)
    axes[1].set_xlabel("Training seed")
    axes[1].set_ylabel("Ratio between Bertrand and monopoly")
    axes[1].set_title("Supra-Bertrand learning is clear, but not full monopoly")
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    return fig


def benchmark_table(bench: Benchmarks, params: Params) -> pd.DataFrame:
    """One-row table of static benchmarks and training size."""
    competitive_profit = float(profits(bench.p_bertrand, params).mean())
    monopoly_profit = float(profits(bench.p_monopoly, params).mean())
    return pd.DataFrame([{
        "Bertrand price": bench.bertrand_price,
        "Monopoly price": bench.monopoly_price,
        "Competitive profit": competitive_profit,
        "Monopoly profit": monopoly_profit,
        "Grid size": params.k,
        "Training steps": params.steps,
    }])


def seed_table(runs: list[SeedRun]) -> pd.DataFrame:
    """Seed-level learned price and deviation diagnostics."""
    rows = []
    for run in runs:
        rows.append({
            "Seed": run.seed,
            "Learned average price": run.learned_price,
            "Learned profit": run.learned_profit,
            "Collusion index": run.collusion_index,
            "Minimum post-deviation price": run.min_post_deviation_price,
            "Recovery horizon": -1 if run.recovery_horizon is None else run.recovery_horizon,
        })
    return pd.DataFrame(rows)


def format_seed_result(runs: list[SeedRun]) -> str:
    """Short prose summary for the generated README."""
    avg_index = float(np.mean([run.collusion_index for run in runs]))
    min_index = float(np.min([run.collusion_index for run in runs]))
    max_index = float(np.max([run.collusion_index for run in runs]))
    horizons = [
        "not recovered" if run.recovery_horizon is None else str(run.recovery_horizon)
        for run in runs
    ]
    min_post = float(np.min([run.min_post_deviation_price for run in runs]))
    return (
        f"Across the five fixed seeds, the mean collusion index is {avg_index:.2f}; "
        f"the range is {min_index:.2f} to {max_index:.2f}. The lowest post-deviation "
        f"average price is {min_post:.3f}. Recovery horizons after the forced "
        f"undercut are {', '.join(horizons)} periods."
    )


def main() -> None:
    setup_style()
    params = Params()
    seeds = [101, 202, 303, 404, 505]
    bench = solve_benchmarks(params)
    runs = [summarize_seed(seed, bench, params) for seed in seeds]
    benchmarks = benchmark_table(bench, params)
    seed_summary = seed_table(runs)

    print("Algorithmic collusion by Q-learning")
    print(f"  Bertrand price = {bench.bertrand_price:.3f}")
    print(f"  Monopoly price = {bench.monopoly_price:.3f}")
    print(f"  Mean learned price = {seed_summary['Learned average price'].mean():.3f}")
    print(f"  Mean collusion index = {seed_summary['Collusion index'].mean():.3f}")

    report = ModelReport(
        "Algorithmic Collusion by Q-Learning",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Algorithmic pricing turns a repeated oligopoly problem into a learning "
        "problem. Two firms choose prices again and again. They do not solve the "
        "dynamic game. They only observe the profit from the price they chose and "
        "update a table of action values.\n\n"
        "The economic question is whether this feedback can move prices above the "
        "static Bertrand-Nash benchmark. In a one-shot differentiated-products "
        "Bertrand game, each firm sets a price that is a best response to the "
        "rival's price. Joint monopoly gives the upper benchmark because one owner "
        "would internalize substitution between the two products.\n\n"
        "This tutorial is deliberately smaller than the Calvano, Calzolari, "
        "Denicolo, and Pastorello experiment and the Courthoud replication code. "
        "It keeps the same model class: logit demand, a finite price grid, and "
        "independent tabular Q-learning. With a short run and five seeds, the "
        "clearest result is supra-Bertrand pricing. The deviation experiment is "
        "more mixed, so the text treats price-war discipline as weak and "
        "seed-dependent rather than as a guaranteed finding."
    )

    report.add_equations(
        r"""
There are two firms, indexed by $i = 1,2$. Firm $i$ chooses price $p_i$ and has
constant marginal cost $c$. Product quality is $a$, the outside-option value is
$a_0$, and $\mu$ controls product differentiation. Logit demand is

$$s_i(p) = \frac{\exp((a - p_i) / \mu)}{\exp(a_0 / \mu) + \sum_{j=1}^2 \exp((a - p_j) / \mu)}.$$

Current profit is

$$\pi_i(p) = (p_i - c)s_i(p).$$

The static Bertrand-Nash price solves each firm's first-order condition,

$$1 - \frac{(p_i - c)(1 - s_i(p))}{\mu} = 0.$$

The joint-monopoly price solves the two-product owner's first-order condition,

$$1 - \frac{(p_i - c)(1 - s_i(p))}{\mu} + \frac{(p_j - c)s_j(p)}{\mu} = 0,\quad j \ne i.$$

The Q-learning state is the previous-period price-index pair
$s_t = (a_{1,t-1}, a_{2,t-1})$. Firm $i$'s action is its current price-grid
index $a_{i,t}$. After observing current profit and next state $s_{t+1}$,
the tabular update is

$$Q_i(s_t, a_{i,t}) \leftarrow (1-\eta) Q_i(s_t, a_{i,t}) + \eta \left[\pi_i(p_t) + \delta \max_a Q_i(s_{t+1}, a)\right].$$

The reported collusion index is

$$\mathrm{CI} = \frac{\bar p_{\mathrm{learned}} - p_{\mathrm{Bertrand}}}{p_{\mathrm{Monopoly}} - p_{\mathrm{Bertrand}}}.$$
"""
    )

    report.add_model_setup(
        "The grid is centered on the static economic benchmarks. First solve the "
        "Bertrand-Nash and joint-monopoly first-order conditions. Then form "
        f"{params.k - 2} evenly spaced prices between those two prices and add one "
        "padding point below and above. The padding point below Bertrand is used "
        "for the forced undercut in the deviation diagnostic.\n\n"
        f"| Object | Value | Role |\n"
        f"|---|---:|---|\n"
        f"| Product value $a$ | {params.a:.2f} | Inside-good quality |\n"
        f"| Outside value $a_0$ | {params.a0:.2f} | Outside option utility |\n"
        f"| Differentiation $\\mu$ | {params.mu:.2f} | Smaller values make products closer substitutes |\n"
        f"| Marginal cost $c$ | {params.c:.2f} | Constant production cost |\n"
        f"| Bertrand price | {bench.bertrand_price:.3f} | Static competitive benchmark |\n"
        f"| Monopoly price | {bench.monopoly_price:.3f} | Joint-profit benchmark |\n"
        f"| Price grid size | {params.k} | Discrete action count per firm |\n"
        f"| Training steps per seed | {params.steps:,} | Q-learning updates |\n"
        f"| Discount factor $\\delta$ | {params.delta:.2f} | Value of future profit |\n"
        f"| Learning rate $\\eta$ | {params.eta:.2f} | Q-table update weight |\n"
        f"| Exploration floor | {params.eps_min:.2f} | Late random-action probability |\n\n"
        "The five seeds are fixed so the generated page is reproducible. The "
        "small sample is a teaching experiment, not a full replication exercise."
    )

    report.add_solution_method(
        "The algorithm is independent Q-learning. Each firm treats the rival and "
        "the market state as part of the environment. There is no explicit "
        "collusion constraint and no direct communication.\n\n"
        "```text\n"
        "Algorithm: independent Q-learning in a repeated pricing game\n"
        "Input: price grid A, profit table pi_i(a_1, a_2), discount delta\n"
        "Output: greedy pricing rules for both firms\n\n"
        "1. Set the initial state to the Bertrand grid point for both firms.\n"
        "2. Initialize Q_i(previous prices, own price) with optimistic\n"
        "   discounted average one-period profits.\n"
        "3. For t = 1 to T:\n"
        "   3a. Each firm observes the previous price-index pair s_t.\n"
        "   3b. With probability epsilon_t, choose a random price index.\n"
        "       Otherwise choose an own price with the highest Q_i(s_t, a_i).\n"
        "   3c. The two actions form current prices and current profits.\n"
        "   3d. The next state is the current action pair.\n"
        "   3e. Update each firm's Q table using realized profit plus the\n"
        "       discounted best continuation value at the next state.\n"
        "4. Freeze exploration and roll out greedy play to measure learned prices.\n"
        "5. Force firm 1 to choose the low padding price once, then return both\n"
        "   firms to greedy play and record the post-deviation path.\n"
        "```\n\n"
        "The deviation diagnostic is intentionally mechanical. It asks whether the "
        "learned policy reacts to an undercut by lowering prices and later "
        "recovering. A sharp fall and recovery would look like punishment. A small "
        "or brief response is weaker evidence."
    )

    report.add_results(
        "Greedy play after training is above the Bertrand price in every seed. "
        "The paths do not reach the monopoly benchmark. They sit in the middle of "
        "the benchmark interval, which is enough to show how independent profit "
        "feedback can support supra-Bertrand prices in a repeated pricing "
        "environment."
    )
    report.add_figure(
        "figures/price-paths.png",
        "Learned greedy price paths after Q-learning",
        plot_price_paths(runs, bench, params),
    )

    report.add_results(
        format_seed_result(runs)
        + " The forced undercut does trigger lower prices in several seeds, but "
        "the response is brief and not uniform. In this reduced tutorial, "
        "supra-Bertrand learning is more robust than price-war-style discipline."
    )
    report.add_figure(
        "figures/deviation-response.png",
        "Forced one-firm low-price deviation followed by greedy play",
        plot_deviation_response(runs, bench),
    )

    report.add_results(
        "The seed diagnostics put the price and profit results on the same scale. "
        "Zero is the Bertrand benchmark and one is the joint-monopoly benchmark. "
        "The price index is consistently positive, while the profit ratio is a "
        "little higher because even moderate price increases raise margins in "
        "this small logit market."
    )
    report.add_figure(
        "figures/learning-diagnostics.png",
        "Seed-level learned price and profit ratios",
        plot_learning_diagnostics(runs, bench),
    )

    report.add_table(
        "tables/benchmark-summary.csv",
        "Static benchmark summary",
        benchmarks,
        "The Bertrand and monopoly prices are solved from the continuous-price "
        "first-order conditions before the finite action grid is built.",
    )

    report.add_table(
        "tables/seed-summary.csv",
        "Seed-level Q-learning outcomes",
        seed_summary,
        "A recovery horizon of -1 means the average price did not return to 95 "
        "percent of the pre-deviation price within the plotted diagnostic window.",
    )

    report.add_takeaway(
        "The small experiment delivers the main teaching result: Q-learning "
        "pricing agents can learn prices above the static Bertrand benchmark "
        "without solving the repeated game. The price-war diagnostic is more "
        "qualified. Some seeds show a short price decline after an undercut, but "
        "the response is not a stable punishment regime in this reduced setup. "
        "That distinction matters: supra-Bertrand learning appears clearly here; "
        "robust collusive discipline would require a larger and more careful "
        "replication."
    )

    report.add_references([
        "[Calvano, E., Calzolari, G., Denicolo, V., and Pastorello, S. (2020). Artificial Intelligence, Algorithmic Pricing, and Collusion. *American Economic Review*, 110(10), 3267-3297.](https://www.aeaweb.org/articles?id=10.1257/aer.20190623)",
        "[Matteo Courthoud. Algorithmic Collusion Replication. GitHub repository.](https://github.com/matteocourthoud/Algorithmic-Collusion-Replication)",
    ])

    report.write("README.md")


if __name__ == "__main__":
    main()
