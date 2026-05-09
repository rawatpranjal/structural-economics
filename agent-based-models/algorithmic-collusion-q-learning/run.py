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

    alpha: float = 0.15
    beta: float = 4e-6
    delta: float = 0.95
    mu: float = 0.25
    a: float = 2.0
    a0: float = 0.0
    c: float = 1.0
    k: int = 15
    n: int = 2
    steps: int = 250_000
    eval_periods: int = 120
    eval_burn: int = 60
    impulse_horizon: int = 30


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
class LearningRun:
    """Learned policy diagnostics for one fixed training run."""

    seed: int
    q: np.ndarray
    greedy_actions: np.ndarray
    greedy_prices: np.ndarray
    greedy_profits: np.ndarray
    learned_price: float
    learned_profit: float
    collusion_index: float
    impulse_prices: np.ndarray
    pre_shock_price: float
    min_post_shock_price: float
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
    if params.n != 2:
        raise ValueError("This teaching implementation is calibrated for two firms.")

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
    state = np.zeros(2, dtype=int)

    for t in range(params.steps):
        epsilon = np.exp(-params.beta * t)
        action = np.empty(2, dtype=int)
        for firm in range(2):
            if rng.random() < epsilon:
                action[firm] = int(rng.integers(params.k))
            else:
                action[firm] = int(np.argmax(q[firm, state[0], state[1]]))

        reward = bench.profit_table[action[0], action[1]]
        next_state = action.copy()
        for firm in range(2):
            old_value = q[firm, state[0], state[1], action[firm]]
            continuation = q[firm, next_state[0], next_state[1]].max()
            target = reward[firm] + params.delta * continuation
            q[firm, state[0], state[1], action[firm]] = (
                (1.0 - params.alpha) * old_value + params.alpha * target
            )
        state = next_state

    return q


def greedy_rollout(q: np.ndarray, bench: Benchmarks, periods: int, start: np.ndarray | None = None) -> np.ndarray:
    """Roll out the learned greedy policy."""
    if start is None:
        state = np.zeros(2, dtype=int)
    else:
        state = np.asarray(start, dtype=int).copy()
    actions = np.empty((periods, 2), dtype=int)
    for t in range(periods):
        action = greedy_action(q, state)
        actions[t] = action
        state = action
    return actions


def impulse_response(q: np.ndarray, bench: Benchmarks, params: Params) -> tuple[np.ndarray, float, float, int | None]:
    """Force firm 1 to undercut once, then roll out the frozen greedy policy."""
    pre_actions = greedy_rollout(q, bench, 40)
    pre_state = pre_actions[-1]
    pre_price = float(bench.grid[pre_actions[-20:]].mean())

    first_action = np.array([
        bench.low_deviation_index,
        int(np.argmax(q[1, pre_state[0], pre_state[1]])),
    ])
    actions = [first_action]
    state = first_action.copy()
    for _ in range(params.impulse_horizon):
        action = greedy_action(q, state)
        actions.append(action)
        state = action

    impulse_prices = bench.grid[np.asarray(actions)]
    average_prices = impulse_prices.mean(axis=1)
    min_post = float(average_prices[1:].min())
    recovered = np.flatnonzero(average_prices[1:] >= 0.95 * pre_price)
    recovery_horizon = int(recovered[0] + 1) if recovered.size else None
    return impulse_prices, pre_price, min_post, recovery_horizon


def summarize_run(seed: int, bench: Benchmarks, params: Params) -> LearningRun:
    """Train and evaluate one fixed Q-learning run."""
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
    impulse_prices, pre_price, min_post, horizon = impulse_response(q, bench, params)
    return LearningRun(
        seed=seed,
        q=q,
        greedy_actions=actions,
        greedy_prices=prices,
        greedy_profits=period_profits,
        learned_price=learned_price,
        learned_profit=learned_profit,
        collusion_index=float(collusion_index),
        impulse_prices=impulse_prices,
        pre_shock_price=pre_price,
        min_post_shock_price=min_post,
        recovery_horizon=horizon,
    )


def plot_price_paths(run: LearningRun, bench: Benchmarks, params: Params) -> plt.Figure:
    """Greedy learned price paths for the fixed run."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    time = np.arange(params.eval_periods)
    avg_price = run.greedy_prices.mean(axis=1)
    ax.plot(time, run.greedy_prices[:, 0], color="C3", linewidth=1.4, alpha=0.75, label="firm 1")
    ax.plot(time, run.greedy_prices[:, 1], color="C0", linewidth=1.4, alpha=0.75, label="firm 2")
    ax.plot(time, avg_price, color="black", linewidth=2.4, label="average")
    ax.axhline(bench.bertrand_price, color="black", linestyle=":", linewidth=1.2, label="Bertrand-Nash")
    ax.axhline(bench.monopoly_price, color="black", linestyle="--", linewidth=1.2, label="Joint monopoly")
    ax.set_xlabel("Greedy play period after training")
    ax.set_ylabel("Price")
    ax.set_title(f"Fixed seed {run.seed}: learned prices above Bertrand")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    return fig


def plot_impulse_response(run: LearningRun, bench: Benchmarks) -> plt.Figure:
    """Impulse response to a forced one-period undercut."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    horizon = run.impulse_prices.shape[0]
    time = np.arange(horizon)
    avg_price = run.impulse_prices.mean(axis=1)

    ax.plot(time, run.impulse_prices[:, 0], color="C3", linewidth=2.0, label="firm 1 price")
    ax.plot(time, run.impulse_prices[:, 1], color="C0", linewidth=2.0, label="firm 2 price")
    ax.plot(time, avg_price, color="black", linewidth=2.5, label="average price")
    ax.axhline(bench.bertrand_price, color="black", linestyle=":", linewidth=1.2, label="Bertrand-Nash")
    ax.axhline(bench.monopoly_price, color="black", linestyle="--", linewidth=1.2, label="Joint monopoly")
    ax.axvline(0, color="0.4", linestyle="-.", linewidth=1.0, label="one-period undercut")
    ax.set_xlabel("Periods after price-deviation shock")
    ax.set_ylabel("Price")
    ax.set_title("Impulse response to one forced low-price action")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    return fig


def plot_learning_diagnostics(run: LearningRun, bench: Benchmarks) -> plt.Figure:
    """Single-run learned price and payoff diagnostics."""
    competitive_profit = bench.profit_table[bench.bertrand_index, bench.bertrand_index].mean()
    monopoly_index = int(np.argmin(np.abs(bench.grid - bench.monopoly_price)))
    monopoly_profit = bench.profit_table[monopoly_index, monopoly_index].mean()
    profit_ratio = (run.learned_profit - competitive_profit) / (monopoly_profit - competitive_profit)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    price_labels = ["Bertrand", f"learned\nseed {run.seed}", "monopoly"]
    price_values = [bench.bertrand_price, run.learned_price, bench.monopoly_price]
    axes[0].bar(np.arange(3), price_values, color=["0.35", "C0", "0.65"], alpha=0.9)
    axes[0].set_xticks(np.arange(3))
    axes[0].set_xticklabels(price_labels)
    axes[0].set_ylabel("Average price")
    axes[0].set_title("Learned price between static benchmarks")

    width = 0.36
    x = np.array([0])
    axes[1].bar(x - width / 2, [run.collusion_index], width, color="C2", label="Price collusion index")
    axes[1].bar(x + width / 2, [profit_ratio], width, color="C4", label="Profit ratio")
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"seed {run.seed}"])
    axes[1].set_ylabel("Ratio between Bertrand and monopoly")
    axes[1].set_title("Single-run normalized diagnostics")
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


def run_table(run: LearningRun) -> pd.DataFrame:
    """Single-run learned price and impulse-response diagnostics."""
    return pd.DataFrame([{
        "Seed": run.seed,
        "Learned average price": run.learned_price,
        "Learned profit": run.learned_profit,
        "Collusion index": run.collusion_index,
        "Pre-shock average price": run.pre_shock_price,
        "Minimum post-shock average price": run.min_post_shock_price,
        "Recovery horizon": -1 if run.recovery_horizon is None else run.recovery_horizon,
    }])


def format_run_result(run: LearningRun) -> str:
    """Short prose summary for the generated README."""
    horizon = "not recovered" if run.recovery_horizon is None else f"{run.recovery_horizon} periods"
    return (
        f"In the fixed seed {run.seed} run, the learned average price is "
        f"{run.learned_price:.3f}. The collusion index is {run.collusion_index:.2f}, "
        f"so the greedy policy sits about halfway between the Bertrand and monopoly "
        f"benchmarks. After the one-period price-deviation shock, the lowest "
        f"post-shock average price is {run.min_post_shock_price:.3f}; the path "
        f"returns to 95 percent of its pre-shock level after {horizon}."
    )


def main() -> None:
    setup_style()
    params = Params()
    seed = 202
    bench = solve_benchmarks(params)
    run = summarize_run(seed, bench, params)
    benchmarks = benchmark_table(bench, params)
    run_summary = run_table(run)

    print("Algorithmic collusion by Q-learning")
    print(f"  Bertrand price = {bench.bertrand_price:.3f}")
    print(f"  Monopoly price = {bench.monopoly_price:.3f}")
    print(f"  Seed = {seed}")
    print(f"  Learned price = {run.learned_price:.3f}")
    print(f"  Collusion index = {run.collusion_index:.3f}")

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
        "It keeps the same model class and moves the main hyperparameters toward "
        "the Courthoud replication defaults: logit demand, a finite price grid, "
        "Courthoud's exponential exploration rule, and independent tabular "
        "Q-learning. The page follows one compact calibrated run with seed "
        f"{seed}. It is not a robustness exercise."
    )

    report.add_equations(
        r"""
There are two firms, indexed by $i = 1,2$. Firm $i$ chooses price $p_i$ and has
constant marginal cost $c$. Product quality is $a$, the outside-option value is
$a_0$, and $\mu$ controls product differentiation. The inside utility index is

$$u_i = \frac{\{a - p_i\}}{\{\mu\}}, \qquad u_0 = \frac{\{a_0\}}{\{\mu\}}.$$

The braces mark the numerator and denominator of each utility index. A lower
price raises $u_i$; a larger $\mu$ makes a given price difference matter less.
Logit demand is

$$s_i(p) = \frac{\exp(u_i)}{\exp(u_0) + \sum_{j=1}^2 \exp(u_j)}.$$

The numerator is product $i$'s exponentiated utility. The denominator is the
outside-good term plus the exponentiated utilities of the two inside goods.

Current profit is

$$\pi_i(p) = (p_i - c)s_i(p).$$

The own-price derivative of the logit share is

$$\frac{\partial s_i}{\partial p_i} = -\frac{s_i(p)(1-s_i(p))}{\mu}.$$

The static Bertrand-Nash price sets $\partial \pi_i / \partial p_i = 0$:

$$\frac{\partial \pi_i}{\partial p_i} = s_i(p) + (p_i-c)\frac{\partial s_i}{\partial p_i} = s_i(p)[1 - \frac{(p_i-c)(1-s_i(p))}{\mu}] = 0.$$

Since $s_i(p)>0$, the Bertrand first-order condition is

$$1 - \frac{(p_i - c)(1 - s_i(p))}{\mu} = 0.$$

The joint monopolist maximizes $\Pi(p)=\pi_1(p)+\pi_2(p)$. Its condition for
product $i$ keeps the Bertrand own-profit term and adds the cross-product term:

$$1 - \frac{(p_i - c)(1 - s_i(p))}{\mu} + \frac{(p_j - c)s_j(p)}{\mu} = 0,\quad j \ne i.$$

The price grid uses the static benchmarks. Let $p_B$ be the Bertrand price,
$p_M$ be the monopoly price, and $\Delta$ be the grid step. The action set is

$$\mathcal{P} = \{p_B-\Delta\} \cup \{p_B, p_B+\Delta,\dots,p_M\} \cup \{p_M+\Delta\}.$$

The Q-learning state is the previous-period price-index pair
$s_t = (a_{1,t-1}, a_{2,t-1})$ (here $s_t$ is the Q-learning state pair, distinct from the demand share $s_i(p)$ defined above). Firm $i$'s action is its current price-grid
index $a_{i,t}$ (where $a_{i,t}$ is a price-grid index, not the product quality parameter $a$ defined above). After observing current profit and next state $s_{t+1}$,
the tabular update is

$$Q_i(s_t, a_{i,t}) \leftarrow (1-\alpha) Q_i(s_t, a_{i,t}) + \alpha [\pi_i(p_t) + \delta \max_a Q_i(s_{t+1}, a)].$$

The reported collusion index is

$$\mathrm{CI} = \frac{\bar p_{\mathrm{learned}} - p_{\mathrm{Bertrand}}}{p_{\mathrm{Monopoly}} - p_{\mathrm{Bertrand}}}.$$
"""
    )

    report.add_model_setup(
        "The grid is centered on the static economic benchmarks. First solve the "
        "Bertrand-Nash and joint-monopoly first-order conditions. Then form "
        f"{params.k - 2} evenly spaced prices between those two prices and add one "
        "padding point below and above. The padding point below Bertrand is the "
        "one-period undercut in the impulse-response diagnostic.\n\n"
        f"| Object | Value | Role |\n"
        f"|---|---:|---|\n"
        f"| Firms $n$ | {params.n} | Symmetric sellers |\n"
        f"| Product value $a$ | {params.a:.2f} | Inside-good quality |\n"
        f"| Outside value $a_0$ | {params.a0:.2f} | Outside option utility |\n"
        f"| Differentiation $\\mu$ | {params.mu:.2f} | Smaller values make products closer substitutes |\n"
        f"| Marginal cost $c$ | {params.c:.2f} | Constant production cost |\n"
        f"| Bertrand price | {bench.bertrand_price:.3f} | Static competitive benchmark |\n"
        f"| Monopoly price | {bench.monopoly_price:.3f} | Joint-profit benchmark |\n"
        f"| Price grid size | {params.k} | Discrete action count per firm |\n"
        f"| Training seed | {seed} | Fixed calibrated run |\n"
        f"| Training steps | {params.steps:,} | Q-learning updates |\n"
        f"| Discount factor $\\delta$ | {params.delta:.2f} | Value of future profit |\n"
        f"| Learning rate $\\alpha$ | {params.alpha:.2f} | Q-table update weight |\n"
        f"| Exploration decay $\\beta$ | {params.beta:.0e} | $\\Pr(\\text{{explore at }}t)=\\exp(-\\beta t)$ |\n\n"
        "These are replication-style hyperparameters, but the computational "
        "budget is intentionally compact. The page reports one fixed run rather "
        "than a multi-seed robustness table."
    )

    report.add_solution_method(
        "The algorithm is independent Q-learning. Each firm treats the rival and "
        "the market state as part of the environment. There is no explicit "
        "collusion constraint and no direct communication.\n\n"
        "```text\n"
        "Algorithm: independent Q-learning in a repeated pricing game\n"
        "Input: price grid A={0,...,k-1}, profit table pi_i(a_1,a_2),\n"
        "       alpha, beta, delta, training length T\n"
        "Output: greedy pricing rules for both firms\n\n"
        "1. Set the initial state to the lowest price-grid point for both firms.\n"
        "2. Initialize Q_i(previous prices, own price) with optimistic\n"
        "   discounted average one-period profits.\n"
        "3. For t = 0 to T-1:\n"
        "   3a. Set epsilon_t = exp(-beta t).\n"
        "   3b. Each firm observes the previous price-index pair s_t.\n"
        "   3c. For each firm i:\n"
        "       with probability epsilon_t, draw a_{i,t} = Uniform({0,...,k-1});\n"
        "       otherwise set a_{i,t} to the first argmax_a Q_i(s_t,a).\n"
        "   3d. Current prices are the grid values indexed by (a_{1,t}, a_{2,t}).\n"
        "   3e. Current profits are pi_i(a_{1,t},a_{2,t}).\n"
        "   3f. Set s_{t+1} = (a_{1,t}, a_{2,t}).\n"
        "   3g. For each firm i, update\n"
        "       Q_i(s_t,a_{i,t}) <- (1-alpha) Q_i(s_t,a_{i,t})\n"
        "       + alpha [ pi_i(a_{1,t},a_{2,t})\n"
        "       + delta max_a Q_i(s_{t+1},a) ].\n"
        "4. Freeze Q and roll out greedy play to measure learned prices.\n"
        "5. For the impulse response, start from the learned greedy state,\n"
        "   set a_{1,0} to the low-grid action once, let firm 2 choose greedily,\n"
        "   then roll out greedy actions from s_1 = (a_{1,0}, a_{2,0}).\n"
        "```\n\n"
        "The impulse response is intentionally mechanical. It asks what the frozen "
        "policy does after a single undercut. The figure is a diagnostic for this "
        "one learned policy, not proof of robust punishment."
    )

    report.add_results(
        f"Greedy play after training is above the Bertrand price in the fixed "
        f"seed {seed} run. The learned path does not reach the monopoly benchmark. "
        "It sits in the middle of the benchmark interval, which is enough for the "
        "teaching point: independent profit feedback can support supra-Bertrand "
        "prices in a repeated pricing environment."
    )
    report.add_figure(
        "figures/price-paths.png",
        "Learned greedy price paths after Q-learning",
        plot_price_paths(run, bench, params),
    )

    report.add_results(
        format_run_result(run)
        + " Read this as an impulse response to a price-deviation shock. The "
        "single run shows how the frozen policy reacts after one forced undercut, "
        "but it does not establish robust price-war discipline."
    )
    report.add_figure(
        "figures/impulse-response.png",
        "Impulse response to a one-period price-deviation shock",
        plot_impulse_response(run, bench),
    )

    report.add_results(
        "The diagnostics put the price and profit results on the same scale. "
        "Zero is the Bertrand benchmark and one is the joint-monopoly benchmark. "
        "The price index is positive in this run, while the profit ratio is a "
        "little higher because moderate price increases raise margins in this "
        "small logit market."
    )
    report.add_figure(
        "figures/learning-diagnostics.png",
        "Single-run learned price and profit ratios",
        plot_learning_diagnostics(run, bench),
    )

    report.add_table(
        "tables/benchmark-summary.csv",
        "Static benchmark summary",
        benchmarks,
        "The Bertrand and monopoly prices are solved from the continuous-price "
        "first-order conditions before the finite action grid is built.",
    )

    report.add_table(
        "tables/run-summary.csv",
        "Single-run Q-learning outcomes",
        run_summary,
        "A recovery horizon of -1 means the average price did not return to 95 "
        "percent of the pre-shock price within the plotted impulse-response window.",
    )

    report.add_takeaway(
        "The small experiment delivers the main teaching result: Q-learning "
        "pricing agents can learn prices above the static Bertrand benchmark "
        "without solving the repeated game. The impulse response is more "
        "qualified. It shows the reaction of one frozen learned policy to one "
        "forced undercut. That distinction matters: supra-Bertrand learning "
        "appears clearly here; robust collusive discipline would require a larger "
        "and more careful replication."
    )

    report.add_references([
        "[Calvano, E., Calzolari, G., Denicolo, V., and Pastorello, S. (2020). Artificial Intelligence, Algorithmic Pricing, and Collusion. *American Economic Review*, 110(10), 3267-3297.](https://www.aeaweb.org/articles?id=10.1257/aer.20190623)",
        "[Matteo Courthoud. Algorithmic Collusion Replication. GitHub repository.](https://github.com/matteocourthoud/Algorithmic-Collusion-Replication)",
    ])

    report.write("README.md")


if __name__ == "__main__":
    main()
