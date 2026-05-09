#!/usr/bin/env python3
"""Zero-intelligence traders in a continuous double auction.

The tutorial simulates budget-constrained random bidding and asking, compares
the realized allocation with the competitive surplus benchmark, and uses simple
price-quantity panels to show why thin double-auction data make slope recovery
fragile.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


@dataclass(frozen=True)
class MarketConfig:
    """Primitives for one zero-intelligence double-auction market."""

    n_traders: int = 10
    demand_intercept: float = 108.0
    supply_intercept: float = 22.0
    demand_slope: float = 4.5
    supply_slope: float = 4.5
    value_noise: float = 3.0
    cost_noise: float = 3.0
    max_price: float = 130.0
    max_events: int = 1_200


def draw_private_values_and_costs(
    config: MarketConfig,
    rng: np.random.Generator,
    demand_shift: float = 0.0,
    supply_shift: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one market's private values and costs from noisy linear schedules."""
    ranks = np.arange(1, config.n_traders + 1)
    values = (
        config.demand_intercept
        + demand_shift
        - config.demand_slope * ranks
        + rng.normal(0.0, config.value_noise, config.n_traders)
    )
    costs = (
        config.supply_intercept
        + supply_shift
        + config.supply_slope * ranks
        + rng.normal(0.0, config.cost_noise, config.n_traders)
    )

    rng.shuffle(values)
    rng.shuffle(costs)
    return values, costs


def competitive_benchmark(values: np.ndarray, costs: np.ndarray) -> dict[str, object]:
    """Compute the efficient double-auction allocation by sorting values and costs."""
    sorted_values = np.sort(values)[::-1]
    sorted_costs = np.sort(costs)
    gains = sorted_values - sorted_costs
    efficient_quantity = int(np.sum(gains > 0.0))
    max_surplus = float(np.sum(gains[:efficient_quantity]))

    if efficient_quantity > 0:
        price_low = float(sorted_costs[efficient_quantity - 1])
        price_high = float(sorted_values[efficient_quantity - 1])
        benchmark_price = 0.5 * (price_low + price_high)
    else:
        price_low = np.nan
        price_high = np.nan
        benchmark_price = np.nan

    return {
        "sorted_values": sorted_values,
        "sorted_costs": sorted_costs,
        "efficient_quantity": efficient_quantity,
        "max_surplus": max_surplus,
        "price_low": price_low,
        "price_high": price_high,
        "benchmark_price": benchmark_price,
    }


def simulate_double_auction(
    values: np.ndarray,
    costs: np.ndarray,
    rng: np.random.Generator,
    config: MarketConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a continuous double auction with constrained random bids and asks."""
    n_traders = len(values)
    active_buyers = np.ones(n_traders, dtype=bool)
    active_sellers = np.ones(n_traders, dtype=bool)
    bids: dict[int, float] = {}
    asks: dict[int, float] = {}
    trades: list[dict[str, float | int | str]] = []
    history: list[dict[str, object]] = []
    cumulative_surplus = 0.0
    last_price = np.nan

    for event in range(1, config.max_events + 1):
        if not active_buyers.any() or not active_sellers.any():
            break

        if rng.random() < 0.5:
            buyer = int(rng.choice(np.flatnonzero(active_buyers)))
            quote = float(rng.uniform(0.0, values[buyer]))
            bids[buyer] = quote
            side = "buyer"
        else:
            seller = int(rng.choice(np.flatnonzero(active_sellers)))
            quote = float(rng.uniform(costs[seller], config.max_price))
            asks[seller] = quote
            side = "seller"

        traded = False
        if bids and asks:
            best_buyer, best_bid = max(bids.items(), key=lambda item: item[1])
            best_seller, best_ask = min(asks.items(), key=lambda item: item[1])
            if best_bid >= best_ask:
                price = 0.5 * (best_bid + best_ask)
                surplus = float(values[best_buyer] - costs[best_seller])
                cumulative_surplus += surplus
                last_price = price
                traded = True

                trades.append({
                    "trade": len(trades) + 1,
                    "event": event,
                    "buyer": best_buyer,
                    "seller": best_seller,
                    "buyer value": values[best_buyer],
                    "seller cost": costs[best_seller],
                    "accepted bid": best_bid,
                    "accepted ask": best_ask,
                    "price": price,
                    "surplus": surplus,
                    "cumulative surplus": cumulative_surplus,
                })

                active_buyers[best_buyer] = False
                active_sellers[best_seller] = False
                bids = {idx: bid for idx, bid in bids.items() if active_buyers[idx]}
                asks = {idx: ask for idx, ask in asks.items() if active_sellers[idx]}

        history.append({
            "event": event,
            "side": side,
            "quote": quote,
            "best_bid": max(bids.values()) if bids else np.nan,
            "best_ask": min(asks.values()) if asks else np.nan,
            "bid_book": tuple(bids.values()),
            "ask_book": tuple(asks.values()),
            "traded": traded,
            "trade_count": len(trades),
            "last_price": last_price,
            "cumulative_surplus": cumulative_surplus,
        })

        if not active_buyers.any() or not active_sellers.any():
            break

    return pd.DataFrame(trades), pd.DataFrame(history)


def ols_line(quantity: np.ndarray, price: np.ndarray) -> tuple[float, float]:
    """Return intercept and slope for price = intercept + slope * quantity."""
    x = np.asarray(quantity, dtype=float)
    y = np.asarray(price, dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(intercept), float(slope)


def run_price_quantity_panel(
    n_traders: int,
    seed: int,
    markets: int = 40,
    shift_width: float = 20.0,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Build demand-shift and supply-shift panels using only market P-Q outcomes."""
    config = MarketConfig(n_traders=n_traders)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []

    for panel, demand_shift, supply_shift in [
        ("demand", 0.0, None),
        ("supply", None, 0.0),
    ]:
        shifts = np.linspace(-shift_width, shift_width, markets)
        for shift in shifts:
            current_demand_shift = shift if demand_shift is None else demand_shift
            current_supply_shift = shift if supply_shift is None else supply_shift
            values, costs = draw_private_values_and_costs(
                config,
                rng,
                demand_shift=current_demand_shift,
                supply_shift=current_supply_shift,
            )
            trades, _ = simulate_double_auction(values, costs, rng, config)
            if trades.empty:
                continue
            benchmark = competitive_benchmark(values, costs)
            rows.append({
                "panel": panel,
                "N": n_traders,
                "shift": shift,
                "quantity": int(len(trades)),
                "mean price": float(trades["price"].mean()),
                "max surplus": float(benchmark["max_surplus"]),
                "realized surplus": float(trades["surplus"].sum()),
            })

    data = pd.DataFrame(rows)
    estimates = {
        "demand": ols_line(
            data.loc[data["panel"] == "demand", "quantity"].to_numpy(),
            data.loc[data["panel"] == "demand", "mean price"].to_numpy(),
        ),
        "supply": ols_line(
            data.loc[data["panel"] == "supply", "quantity"].to_numpy(),
            data.loc[data["panel"] == "supply", "mean price"].to_numpy(),
        ),
    }
    return data, estimates


def save_order_book_gif(history: pd.DataFrame, max_price: float, path: str) -> None:
    """Animate best outstanding bids and asks during the double auction."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n_frames = min(70, len(history))
    frame_idx = np.linspace(0, len(history) - 1, n_frames).astype(int)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    def update(frame_number: int) -> None:
        row = history.iloc[frame_idx[frame_number]]
        ax.clear()
        bid_book = np.array(row["bid_book"], dtype=float)
        ask_book = np.array(row["ask_book"], dtype=float)

        if bid_book.size:
            ax.scatter(
                np.full(bid_book.size, 0.85),
                bid_book,
                s=45,
                alpha=0.7,
                color="#1f77b4",
                label="bids",
            )
        if ask_book.size:
            ax.scatter(
                np.full(ask_book.size, 1.15),
                ask_book,
                s=45,
                alpha=0.7,
                color="#d62728",
                label="asks",
            )
        if not np.isnan(row["last_price"]):
            ax.axhline(row["last_price"], color="black", linestyle=":", linewidth=1.3)

        ax.set_xlim(0.55, 1.45)
        ax.set_ylim(0.0, max_price)
        ax.set_xticks([0.85, 1.15])
        ax.set_xticklabels(["Bids", "Asks"])
        ax.set_ylabel("Price")
        ax.set_title("Random Constrained Orders")
        ax.text(
            0.02,
            0.96,
            f"event {int(row['event'])} | trades {int(row['trade_count'])}",
            transform=ax.transAxes,
            va="top",
        )
        if row["traded"]:
            ax.text(0.02, 0.86, "trade clears", transform=ax.transAxes, va="top")
        if bid_book.size or ask_book.size:
            ax.legend(loc="lower right")

    animation = FuncAnimation(fig, update, frames=n_frames, interval=250)
    animation.save(path, writer=PillowWriter(fps=4))
    plt.close(fig)


def save_convergence_gif(
    trades: pd.DataFrame,
    max_surplus: float,
    benchmark_price: float,
    path: str,
) -> None:
    """Animate transaction prices and cumulative allocative efficiency."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n_frames = max(2, len(trades))
    trade_numbers = trades["trade"].to_numpy()
    prices = trades["price"].to_numpy()
    efficiency = trades["cumulative surplus"].to_numpy() / max_surplus

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))

    def update(frame_number: int) -> None:
        k = min(frame_number + 1, len(trades))
        for ax in axes:
            ax.clear()

        axes[0].plot(trade_numbers[:k], prices[:k], marker="o", color="#1f77b4")
        axes[0].axhline(benchmark_price, color="black", linestyle=":", linewidth=1.2)
        axes[0].set_xlabel("Trade")
        axes[0].set_ylabel("Price")
        axes[0].set_title("Transaction Prices")
        axes[0].set_xlim(1, max(2, len(trades)))

        axes[1].plot(trade_numbers[:k], efficiency[:k], marker="o", color="#2ca02c")
        axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1.2)
        axes[1].set_xlabel("Trade")
        axes[1].set_ylabel("Surplus / max surplus")
        axes[1].set_title("Allocative Efficiency")
        axes[1].set_xlim(1, max(2, len(trades)))
        axes[1].set_ylim(0.0, 1.08)

    animation = FuncAnimation(fig, update, frames=n_frames, interval=450)
    animation.save(path, writer=PillowWriter(fps=3))
    plt.close(fig)


def make_transaction_path_figure(
    trades: pd.DataFrame,
    benchmark_price: float,
) -> plt.Figure:
    """Plot transaction prices and cumulative quantity."""
    fig, ax1 = plt.subplots(figsize=(7, 4.6))
    ax1.plot(trades["trade"], trades["price"], marker="o", color="#1f77b4")
    ax1.axhline(
        benchmark_price,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="competitive midpoint",
    )
    ax1.set_xlabel("Trade")
    ax1.set_ylabel("Transaction price")

    ax2 = ax1.twinx()
    ax2.step(
        trades["trade"],
        trades["trade"],
        where="post",
        color="#2ca02c",
        alpha=0.8,
        label="cumulative quantity",
    )
    ax2.set_ylabel("Cumulative quantity")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    ax1.set_title("Prices and Quantity in One Double-Auction Session")
    fig.tight_layout()
    return fig


def make_allocation_figure(
    values: np.ndarray,
    costs: np.ndarray,
    trades: pd.DataFrame,
    benchmark: dict[str, object],
) -> plt.Figure:
    """Plot sorted surplus opportunities and realized transaction prices."""
    sorted_values = benchmark["sorted_values"]
    sorted_costs = benchmark["sorted_costs"]
    q_grid = np.arange(1, len(sorted_values) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(q_grid, sorted_values, marker="o", label="buyer values", color="#1f77b4")
    ax.plot(q_grid, sorted_costs, marker="o", label="seller costs", color="#d62728")
    ax.scatter(
        trades["trade"],
        trades["price"],
        marker="x",
        s=60,
        color="black",
        label="transaction prices",
    )
    ax.axvline(
        benchmark["efficient_quantity"],
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label="efficient quantity",
    )
    ax.set_xlabel("Unit rank or transaction number")
    ax.set_ylabel("Dollars")
    ax.set_title("Competitive Benchmark and Realized Trades")
    ax.legend(loc="best")
    return fig


def make_slope_figure(panel_data: pd.DataFrame, estimates: pd.DataFrame) -> plt.Figure:
    """Plot price-quantity panels and fitted OLS lines."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharey=True)
    colors = {10: "#1f77b4", 5: "#d62728"}
    titles = {
        "demand": "Supply shifts: demand slope",
        "supply": "Demand shifts: supply slope",
    }

    for ax, panel in zip(axes, ["demand", "supply"]):
        subset_panel = panel_data[panel_data["panel"] == panel]
        for n_traders, group in subset_panel.groupby("N"):
            ax.scatter(
                group["quantity"],
                group["mean price"],
                alpha=0.62,
                s=28,
                color=colors[int(n_traders)],
                label=f"N={int(n_traders)}",
            )
            estimate = estimates[
                (estimates["Panel"] == panel) & (estimates["N"] == int(n_traders))
            ].iloc[0]
            x_vals = np.array([group["quantity"].min(), group["quantity"].max()])
            y_vals = estimate["Intercept"] + estimate["OLS slope"] * x_vals
            ax.plot(x_vals, y_vals, color=colors[int(n_traders)], linewidth=2.0)

        ax.set_title(titles[panel])
        ax.set_xlabel("Realized quantity")
        ax.legend(loc="best")
    axes[0].set_ylabel("Mean transaction price")
    fig.tight_layout()
    return fig


def main() -> None:
    setup_style()
    config = MarketConfig()
    baseline_rng = np.random.default_rng(1)
    values, costs = draw_private_values_and_costs(config, baseline_rng)
    benchmark = competitive_benchmark(values, costs)
    trades, history = simulate_double_auction(values, costs, baseline_rng, config)

    if trades.empty:
        raise RuntimeError("Baseline market produced no trades.")

    max_surplus = float(benchmark["max_surplus"])
    realized_surplus = float(trades["surplus"].sum())
    allocative_efficiency = realized_surplus / max_surplus
    mean_price = float(trades["price"].mean())
    price_path_sd = float(trades["price"].std(ddof=0))

    panel_rows = []
    estimate_rows = []
    true_slopes = {"demand": -config.demand_slope, "supply": config.supply_slope}
    for n_traders in [10, 5]:
        panel_data, panel_estimates = run_price_quantity_panel(n_traders, seed=100)
        panel_rows.append(panel_data)
        for panel, (intercept, slope) in panel_estimates.items():
            estimate_rows.append({
                "N": n_traders,
                "Panel": panel,
                "True slope": true_slopes[panel],
                "Intercept": intercept,
                "OLS slope": slope,
                "Slope error": slope - true_slopes[panel],
                "Abs. error": abs(slope - true_slopes[panel]),
            })

    all_panel_data = pd.concat(panel_rows, ignore_index=True)
    estimates = pd.DataFrame(estimate_rows)

    summary = pd.DataFrame([
        {
            "Object": "Buyers",
            "Value": config.n_traders,
            "Role": "Random private values",
        },
        {
            "Object": "Sellers",
            "Value": config.n_traders,
            "Role": "Random private costs",
        },
        {
            "Object": "Realized trades",
            "Value": len(trades),
            "Role": "Trades cleared by the double-auction institution",
        },
        {
            "Object": "Competitive quantity",
            "Value": int(benchmark["efficient_quantity"]),
            "Role": "Sorted value-cost pairs with positive surplus",
        },
        {
            "Object": "Allocative efficiency",
            "Value": f"{100.0 * allocative_efficiency:.2f}%",
            "Role": "Realized surplus divided by maximum surplus",
        },
        {
            "Object": "Mean transaction price",
            "Value": f"{mean_price:.2f}",
            "Role": "Average midpoint price across realized trades",
        },
    ])

    transaction_log = trades.copy()
    numeric_cols = [
        "buyer value",
        "seller cost",
        "accepted bid",
        "accepted ask",
        "price",
        "surplus",
        "cumulative surplus",
    ]
    transaction_log[numeric_cols] = transaction_log[numeric_cols].round(3)

    Path("tables").mkdir(exist_ok=True)
    transaction_log.to_csv("tables/transaction-log.csv", index=False)
    all_panel_data.to_csv("tables/price-quantity-panel.csv", index=False)

    save_order_book_gif(history, config.max_price, "figures/order-book.gif")
    save_convergence_gif(
        trades,
        max_surplus,
        float(benchmark["benchmark_price"]),
        "figures/convergence.gif",
    )

    report = ModelReport(
        "Zero-Intelligence Traders in a Double Auction",
        "Random constrained orders, competitive surplus, and thin-market slope recovery.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A double auction lets buyers and sellers post prices while the market is "
        "open. Buyers know private values. Sellers know private costs. A trade "
        "clears when the best bid is at least as high as the best ask.\n\n"
        "This tutorial strips behavior down to zero intelligence. Buyers draw "
        "random bids that never exceed value. Sellers draw random asks that never "
        "fall below cost. The agents do not optimize, forecast, or learn.\n\n"
        "The economic question is how much work the market institution does. The "
        "simulation compares random constrained trading with the competitive "
        "surplus benchmark, then asks what can be recovered from price-quantity "
        "data alone."
    )

    report.add_equations(
        r"""
Buyer $i$ has value $v_i$ for one unit.
Seller $j$ has cost $c_j$ for one unit.
A zero-intelligence constrained buyer draws a bid $b_i$ satisfying

$$
0 \leq b_i \leq v_i.
$$

A zero-intelligence constrained seller draws an ask $a_j$ satisfying

$$
c_j \leq a_j \leq \bar p.
$$

The double auction clears when the best bid crosses the best ask:

$$
\max_i b_i \geq \min_j a_j.
$$

The transaction price is the midpoint between the accepted bid and ask.
The realized surplus from a trade between buyer $i$ and seller $j$ is

$$
s_{ij}=v_i-c_j.
$$

The competitive benchmark sorts buyer values from high to low and seller costs
from low to high. With sorted values $v_{(q)}$ and sorted costs $c_{(q)}$, the
efficient quantity is

$$
Q^{\ast}=\max \{q: v_{(q)}-c_{(q)} > 0\}.
$$

Maximum surplus is

$$
S^{\ast}=\sum_{q=1}^{Q^{\ast}} [v_{(q)}-c_{(q)}].
$$

Allocative efficiency is

$$
\mathrm{AE}=\frac{\sum_{\mathrm{trades}} s_{ij}}{S^{\ast}}.
$$

For the slope exercise, each market session produces a realized quantity $Q_m$
and an average transaction price $P_m$. OLS fits

$$
P_m=\alpha+\beta Q_m+\epsilon_m.
$$

Supply-shift panels trace out the demand slope. Demand-shift panels trace out
the supply slope.
"""
    )

    report.add_model_setup(
        "| Object | Value | Role |\n"
        "|---|---:|---|\n"
        f"| Buyers | {config.n_traders} | One-unit demand with private values |\n"
        f"| Sellers | {config.n_traders} | One-unit supply with private costs |\n"
        f"| Value schedule | intercept {config.demand_intercept:.0f}, slope {-config.demand_slope:.1f} | Noisy downward demand |\n"
        f"| Cost schedule | intercept {config.supply_intercept:.0f}, slope {config.supply_slope:.1f} | Noisy upward supply |\n"
        f"| Random order cap | {config.max_price:.0f} | Maximum ask support |\n"
        f"| Quote events | up to {config.max_events:,} | Repeated random bids and asks |\n"
        "| Transaction price | midpoint | Accepted bid and ask split the spread |\n"
        "| OLS panels | 40 sessions per shift design | Uses only realized price and quantity |"
    )

    report.add_solution_method(
        "The simulation has three steps: simulate the institution, solve the "
        "competitive benchmark, and estimate slopes from price-quantity panels.\n\n"
        "```text\n"
        "Algorithm: zero-intelligence constrained double auction\n"
        "Inputs: buyer values v_i, seller costs c_j, price cap pbar\n"
        "Outputs: transaction log, realized surplus, competitive benchmark\n\n"
        "1. Mark every buyer and seller as active.\n"
        "2. Draw a random active trader.\n"
        "3. If the trader is a buyer, draw b_i uniformly from [0, v_i].\n"
        "4. If the trader is a seller, draw a_j uniformly from [c_j, pbar].\n"
        "5. If the highest live bid crosses the lowest live ask, clear one trade.\n"
        "6. Remove the matched buyer and seller and record price, quantity, and surplus.\n"
        "7. Repeat until no active side remains or the quote-event limit is reached.\n"
        "```\n\n"
        "The competitive benchmark is analytical. Sort values from high to low, "
        "sort costs from low to high, and keep the positive value-cost gaps. That "
        "benchmark is the denominator for allocative efficiency.\n\n"
        "The OLS exercise uses repeated market sessions. To recover the demand "
        "slope, the code shifts costs and observes how transaction prices and "
        "quantities move along demand. To recover the supply slope, it shifts "
        "values and moves along supply. The estimator does not see private values "
        "or costs."
    )

    report.add_results(
        f"The baseline market clears {len(trades)} trades with 10 buyers and 10 "
        f"sellers. The competitive benchmark has quantity "
        f"{int(benchmark['efficient_quantity'])}. Realized surplus is "
        f"{realized_surplus:.2f} against a maximum of {max_surplus:.2f}, so "
        f"allocative efficiency is {100.0 * allocative_efficiency:.2f}%. The "
        "agents are random, but the budget constraints and the double-auction "
        "clearing rule keep most high-value buyers matched with low-cost sellers."
    )
    report.add_figure(
        "figures/transaction-path.png",
        "Transaction prices and cumulative quantity in the baseline session.",
        make_transaction_path_figure(trades, float(benchmark["benchmark_price"])),
    )

    report.add_results(
        "The sorted values and costs give the competitive surplus benchmark. "
        "The realized transaction prices sit inside the value-cost range. A few "
        "matches can be out of rank order, but the surplus loss is small in this "
        "session."
    )
    report.add_figure(
        "figures/allocation-benchmark.png",
        "Sorted buyer values, seller costs, and realized transaction prices.",
        make_allocation_figure(values, costs, trades, benchmark),
    )

    report.add_results(
        "The order-book animation shows how little intelligence the agents have. "
        "Quotes arrive randomly. Trades clear only because the institution keeps "
        "the best bid and best ask visible.\n\n"
        '<img src="figures/order-book.gif" alt="Animated double-auction order book." width="80%">\n\n'
        "The convergence animation accumulates realized surplus trade by trade. "
        "The benchmark line is the maximum surplus from the sorted competitive "
        "allocation.\n\n"
        '<img src="figures/convergence.gif" alt="Animated transaction prices and cumulative allocative efficiency." width="80%">'
    )

    report.add_table(
        "tables/baseline-summary.csv",
        "Baseline Market Summary",
        summary,
    )
    report.add_table(
        "tables/transaction-log.csv",
        "Baseline Transaction Log",
        transaction_log[[
            "trade",
            "event",
            "buyer value",
            "seller cost",
            "accepted bid",
            "accepted ask",
            "price",
            "surplus",
        ]],
        description=(
            f"The transaction-price standard deviation is {price_path_sd:.2f}. "
            "The table records the hidden values and costs only for audit; the "
            "slope estimator below uses price and quantity outcomes."
        ),
    )

    display_estimates = estimates.copy()
    for col in ["True slope", "Intercept", "OLS slope", "Slope error", "Abs. error"]:
        display_estimates[col] = display_estimates[col].map(lambda value: f"{value:.3f}")

    report.add_results(
        "The OLS panels use only session-level transaction price and quantity. "
        "With 10 buyers and 10 sellers, the supply-shift panel recovers a "
        "downward demand slope and the demand-shift panel recovers an upward "
        "supply slope. With 5 buyers and 5 sellers, there are too few quantity "
        "points and too much transaction-price noise. The reduced-form estimates "
        "move away from the structural slopes."
    )
    report.add_figure(
        "figures/slope-estimates.png",
        "OLS price-quantity slope recovery in thick and thin markets.",
        make_slope_figure(all_panel_data, estimates),
    )
    report.add_table(
        "tables/slope-estimates.csv",
        "OLS Slope Recovery",
        display_estimates[[
            "N",
            "Panel",
            "True slope",
            "OLS slope",
            "Slope error",
            "Abs. error",
        ]],
    )

    report.add_takeaway(
        "Zero-intelligence constrained traders show how much allocation can come "
        "from the institution rather than from sophisticated strategy. In the "
        "baseline run, random constrained orders still reach about 99% allocative "
        "efficiency.\n\n"
        "The estimation lesson is different. Price-quantity regressions can look "
        "reasonable when the market is thick and the variation is well staged, but "
        "they break quickly when only a few traders generate the time series. That "
        "is the point where a structural model of values, costs, and the trading "
        "institution becomes useful."
    )

    report.add_references([
        "[Gode, D. K. and Sunder, S. (1993). Allocative Efficiency of Markets with Zero-Intelligence Traders: Market as a Partial Substitute for Individual Rationality. *Journal of Political Economy*, 101(1), 119-137.](https://doi.org/10.1086/261868)",
        "[Smith, V. L. (1962). An Experimental Study of Competitive Market Behavior. *Journal of Political Economy*, 70(2), 111-137.](https://doi.org/10.1086/258609)",
        "[Cliff, D. and Bruten, J. (1997). Minimal-intelligence agents for bargaining behaviors in market-based environments. Technical report, Hewlett-Packard Laboratories.](https://www.hpl.hp.com/techreports/97/HPL-97-91.html)",
    ])

    report.write("README.md")
    print(
        "Generated README.md, "
        f"{len(report._figures)} static figures, 2 GIFs, and {len(report._tables)} tables."
    )
    print(f"Baseline allocative efficiency: {100.0 * allocative_efficiency:.2f}%")


if __name__ == "__main__":
    main()
