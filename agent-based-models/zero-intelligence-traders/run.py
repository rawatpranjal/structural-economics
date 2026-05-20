#!/usr/bin/env python3
"""Zero-intelligence traders in a continuous double auction.

The tutorial reproduces the central Gode-Sunder lesson in a small induced-value
market. Budget-constrained random traders recover most available surplus. A
simple ZIP-style adaptive rule mainly tightens prices rather than changing the
allocation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


@dataclass(frozen=True)
class MarketSpec:
    """Induced buyer values and seller costs for one double-auction market."""

    market_type: str
    values: tuple[float, ...]
    costs: tuple[float, ...]
    max_price: float = 125.0
    max_events: int = 2_500

    @property
    def n_buyers(self) -> int:
        return len(self.values)

    @property
    def n_sellers(self) -> int:
        return len(self.costs)


@dataclass(frozen=True)
class LearningConfig:
    """Parameters for the ZIP-style adaptive quote target."""

    learning_rate: float = 0.35
    spread: float = 1.25
    quote_noise: float = 0.9


def market_specs() -> dict[str, MarketSpec]:
    """Return the deterministic stepped markets used in the tutorial."""
    balanced_values = (105, 100, 95, 90, 85, 80, 75, 70, 65, 60)
    balanced_costs = (30, 36, 42, 48, 54, 60, 66, 72, 78, 84)
    return {
        "balanced": MarketSpec(
            "Balanced 10 x 10",
            balanced_values,
            balanced_costs,
        ),
        "buyer_heavy": MarketSpec(
            "Buyer-heavy 15 x 10",
            (113, 108, 103, 98, 93, 88, 83, 78, 73, 68, 63, 58, 53, 48, 43),
            balanced_costs,
        ),
        "seller_heavy": MarketSpec(
            "Seller-heavy 10 x 15",
            balanced_values,
            (24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108),
        ),
        "thin": MarketSpec(
            "Thin 5 x 5",
            (105, 94, 83, 72, 61),
            (35, 46, 57, 68, 79),
        ),
    }


def competitive_benchmark(values: np.ndarray, costs: np.ndarray) -> dict[str, object]:
    """Compute efficient surplus and the competitive price band."""
    sorted_values = np.sort(values)[::-1]
    sorted_costs = np.sort(costs)
    paired_units = min(len(sorted_values), len(sorted_costs))
    gains = sorted_values[:paired_units] - sorted_costs[:paired_units]
    efficient_quantity = int(np.sum(gains > 0.0))
    max_surplus = float(np.sum(gains[:efficient_quantity]))

    if efficient_quantity == 0:
        price_low = np.nan
        price_high = np.nan
    else:
        q = efficient_quantity
        lower_candidates = [float(sorted_costs[q - 1])]
        upper_candidates = [float(sorted_values[q - 1])]
        if q < len(sorted_values):
            lower_candidates.append(float(sorted_values[q]))
        if q < len(sorted_costs):
            upper_candidates.append(float(sorted_costs[q]))
        price_low = max(lower_candidates)
        price_high = min(upper_candidates)

    return {
        "sorted_values": sorted_values,
        "sorted_costs": sorted_costs,
        "gains": gains,
        "efficient_quantity": efficient_quantity,
        "max_surplus": max_surplus,
        "price_low": price_low,
        "price_high": price_high,
        "price_midpoint": 0.5 * (price_low + price_high),
    }


def feasible_random_bid(value: float, rng: np.random.Generator) -> float:
    """Draw a zero-intelligence constrained bid."""
    return float(rng.uniform(0.0, value))


def feasible_random_ask(cost: float, max_price: float, rng: np.random.Generator) -> float:
    """Draw a zero-intelligence constrained ask."""
    return float(rng.uniform(cost, max_price))


def initialize_zip_targets(
    values: np.ndarray,
    costs: np.ndarray,
    benchmark: dict[str, object],
    learning: LearningConfig,
    max_price: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize adaptive quote targets around the competitive band."""
    reference_price = float(benchmark["price_midpoint"])
    buyer_targets = np.minimum(values, reference_price + learning.spread)
    seller_targets = np.maximum(costs, reference_price - learning.spread)
    return (
        np.clip(buyer_targets, 0.0, values),
        np.clip(seller_targets, costs, max_price),
    )


def update_zip_targets(
    buyer_targets: np.ndarray,
    seller_targets: np.ndarray,
    active_buyers: np.ndarray,
    active_sellers: np.ndarray,
    values: np.ndarray,
    costs: np.ndarray,
    last_price: float,
    zip_buyers: set[int],
    zip_sellers: set[int],
    learning: LearningConfig,
    max_price: float,
) -> None:
    """Move active ZIP-style targets toward the latest transaction price."""
    for buyer in zip_buyers:
        if active_buyers[buyer]:
            target = min(values[buyer], last_price + learning.spread)
            buyer_targets[buyer] = (
                (1.0 - learning.learning_rate) * buyer_targets[buyer]
                + learning.learning_rate * target
            )
            buyer_targets[buyer] = float(np.clip(buyer_targets[buyer], 0.0, values[buyer]))

    for seller in zip_sellers:
        if active_sellers[seller]:
            target = max(costs[seller], last_price - learning.spread)
            seller_targets[seller] = (
                (1.0 - learning.learning_rate) * seller_targets[seller]
                + learning.learning_rate * target
            )
            seller_targets[seller] = float(
                np.clip(seller_targets[seller], costs[seller], max_price)
            )


def draw_bid(
    buyer: int,
    values: np.ndarray,
    buyer_targets: np.ndarray,
    zip_buyers: set[int],
    learning: LearningConfig,
    rng: np.random.Generator,
) -> float:
    """Draw either a ZIC bid or a ZIP-style adaptive bid."""
    value = float(values[buyer])
    if buyer not in zip_buyers:
        return feasible_random_bid(value, rng)
    quote = buyer_targets[buyer] + rng.normal(0.0, learning.quote_noise)
    return float(np.clip(quote, 0.0, value))


def draw_ask(
    seller: int,
    costs: np.ndarray,
    seller_targets: np.ndarray,
    zip_sellers: set[int],
    learning: LearningConfig,
    max_price: float,
    rng: np.random.Generator,
) -> float:
    """Draw either a ZIC ask or a ZIP-style adaptive ask."""
    cost = float(costs[seller])
    if seller not in zip_sellers:
        return feasible_random_ask(cost, max_price, rng)
    quote = seller_targets[seller] + rng.normal(0.0, learning.quote_noise)
    return float(np.clip(quote, cost, max_price))


def simulate_double_auction(
    spec: MarketSpec,
    seed: int,
    zip_buyers: set[int] | None = None,
    zip_sellers: set[int] | None = None,
    learning: LearningConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a continuous double auction with constrained quote rules."""
    rng = np.random.default_rng(seed)
    values = np.array(spec.values, dtype=float)
    costs = np.array(spec.costs, dtype=float)
    benchmark = competitive_benchmark(values, costs)
    learning = learning or LearningConfig()
    zip_buyers = zip_buyers or set()
    zip_sellers = zip_sellers or set()

    active_buyers = np.ones(len(values), dtype=bool)
    active_sellers = np.ones(len(costs), dtype=bool)
    buyer_targets, seller_targets = initialize_zip_targets(
        values,
        costs,
        benchmark,
        learning,
        spec.max_price,
    )
    bids: dict[int, float] = {}
    asks: dict[int, float] = {}
    trades: list[dict[str, float | int | str]] = []
    history: list[dict[str, object]] = []
    cumulative_surplus = 0.0
    last_price = np.nan

    for event in range(1, spec.max_events + 1):
        active_buyer_count = int(active_buyers.sum())
        active_seller_count = int(active_sellers.sum())
        if active_buyer_count == 0 or active_seller_count == 0:
            break

        buyer_probability = active_buyer_count / (active_buyer_count + active_seller_count)
        if rng.random() < buyer_probability:
            buyer = int(rng.choice(np.flatnonzero(active_buyers)))
            quote = draw_bid(buyer, values, buyer_targets, zip_buyers, learning, rng)
            bids[buyer] = quote
            side = "buyer"
        else:
            seller = int(rng.choice(np.flatnonzero(active_sellers)))
            quote = draw_ask(
                seller,
                costs,
                seller_targets,
                zip_sellers,
                learning,
                spec.max_price,
                rng,
            )
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
                    "Trade": len(trades) + 1,
                    "Event": event,
                    "Buyer": best_buyer + 1,
                    "Seller": best_seller + 1,
                    "Buyer value": values[best_buyer],
                    "Seller cost": costs[best_seller],
                    "Accepted bid": best_bid,
                    "Accepted ask": best_ask,
                    "Price": price,
                    "Surplus": surplus,
                    "Cumulative surplus": cumulative_surplus,
                })

                active_buyers[best_buyer] = False
                active_sellers[best_seller] = False
                bids = {idx: bid for idx, bid in bids.items() if active_buyers[idx]}
                asks = {idx: ask for idx, ask in asks.items() if active_sellers[idx]}

                update_zip_targets(
                    buyer_targets,
                    seller_targets,
                    active_buyers,
                    active_sellers,
                    values,
                    costs,
                    price,
                    zip_buyers,
                    zip_sellers,
                    learning,
                    spec.max_price,
                )

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

    return pd.DataFrame(trades), pd.DataFrame(history)


def session_metrics(
    spec: MarketSpec,
    trades: pd.DataFrame,
    benchmark: dict[str, object],
) -> dict[str, float | int | str]:
    """Summarize one simulated market session."""
    max_surplus = float(benchmark["max_surplus"])
    realized_surplus = float(trades["Surplus"].sum()) if not trades.empty else 0.0
    price_low = float(benchmark["price_low"])
    price_high = float(benchmark["price_high"])
    if trades.empty:
        mean_price = np.nan
        price_sd = np.nan
        inside_band = np.nan
    else:
        mean_price = float(trades["Price"].mean())
        price_sd = float(trades["Price"].std(ddof=0))
        inside_band = float(
            ((trades["Price"] >= price_low) & (trades["Price"] <= price_high)).mean()
        )

    return {
        "Market type": spec.market_type,
        "Buyers": spec.n_buyers,
        "Sellers": spec.n_sellers,
        "Efficient quantity": int(benchmark["efficient_quantity"]),
        "Competitive price low": price_low,
        "Competitive price high": price_high,
        "Trades": int(len(trades)),
        "Mean price": mean_price,
        "Price SD": price_sd,
        "Allocative efficiency": realized_surplus / max_surplus if max_surplus > 0 else np.nan,
        "Price inside competitive band": inside_band,
    }


def run_market_type_sweep(specs: dict[str, MarketSpec]) -> pd.DataFrame:
    """Simulate the balanced, imbalanced, and thin ZIC markets."""
    rows = []
    seeds = {
        "balanced": 11,
        "buyer_heavy": 12,
        "seller_heavy": 13,
        "thin": 14,
    }
    for key in ["balanced", "buyer_heavy", "seller_heavy", "thin"]:
        spec = specs[key]
        values = np.array(spec.values, dtype=float)
        costs = np.array(spec.costs, dtype=float)
        benchmark = competitive_benchmark(values, costs)
        trades, _ = simulate_double_auction(spec, seed=seeds[key])
        rows.append(session_metrics(spec, trades, benchmark))
    return pd.DataFrame(rows)


def run_agent_mix_sweep(spec: MarketSpec) -> pd.DataFrame:
    """Compare ZIC and ZIP-style quote rules in the same baseline market."""
    values = np.array(spec.values, dtype=float)
    costs = np.array(spec.costs, dtype=float)
    benchmark = competitive_benchmark(values, costs)
    mixes = [
        ("All ZIC", set(), set(), 21),
        ("One ZIP buyer and one ZIP seller", {0}, {0}, 22),
        (
            "All ZIP",
            set(range(spec.n_buyers)),
            set(range(spec.n_sellers)),
            23,
        ),
    ]

    rows = []
    for label, zip_buyers, zip_sellers, seed in mixes:
        trades, _ = simulate_double_auction(
            spec,
            seed=seed,
            zip_buyers=zip_buyers,
            zip_sellers=zip_sellers,
        )
        row = session_metrics(spec, trades, benchmark)
        rows.append({
            "Strategy mix": label,
            "ZIP buyers": len(zip_buyers),
            "ZIP sellers": len(zip_sellers),
            "Trades": row["Trades"],
            "Mean price": row["Mean price"],
            "Price SD": row["Price SD"],
            "Allocative efficiency": row["Allocative efficiency"],
            "Price inside competitive band": row["Price inside competitive band"],
        })
    return pd.DataFrame(rows)


def format_money(value: float) -> str:
    """Format a price-like value for reader-facing tables."""
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}"


def format_percent(value: float) -> str:
    """Format a fraction as a reader-facing percent."""
    if pd.isna(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def display_market_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Prepare the market-type summary table for the report and CSV."""
    table = summary[[
        "Market type",
        "Buyers",
        "Sellers",
        "Efficient quantity",
        "Competitive price low",
        "Competitive price high",
        "Trades",
        "Mean price",
        "Price SD",
        "Allocative efficiency",
    ]].copy()
    for col in ["Competitive price low", "Competitive price high", "Mean price", "Price SD"]:
        table[col] = table[col].map(format_money)
    table["Allocative efficiency"] = table["Allocative efficiency"].map(format_percent)
    return table


def display_agent_mix_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Prepare the agent-mix summary table for the report and CSV."""
    table = summary[[
        "Strategy mix",
        "ZIP buyers",
        "ZIP sellers",
        "Trades",
        "Mean price",
        "Price SD",
        "Allocative efficiency",
        "Price inside competitive band",
    ]].copy()
    for col in ["Mean price", "Price SD"]:
        table[col] = table[col].map(format_money)
    table["Allocative efficiency"] = table["Allocative efficiency"].map(format_percent)
    table["Price inside competitive band"] = table["Price inside competitive band"].map(
        format_percent
    )
    return table


def display_transaction_log(trades: pd.DataFrame) -> pd.DataFrame:
    """Prepare the baseline transaction log with reader-facing labels."""
    cols = [
        "Trade",
        "Event",
        "Buyer value",
        "Seller cost",
        "Accepted bid",
        "Accepted ask",
        "Price",
        "Surplus",
    ]
    table = trades[cols].copy()
    for col in cols[2:]:
        table[col] = table[col].round(3)
    return table


def make_demand_supply_figure(
    values: np.ndarray,
    costs: np.ndarray,
    benchmark: dict[str, object],
) -> plt.Figure:
    """Plot the stepped induced demand and supply schedules."""
    sorted_values = benchmark["sorted_values"]
    sorted_costs = benchmark["sorted_costs"]
    q_values = np.arange(1, len(sorted_values) + 1)
    q_costs = np.arange(1, len(sorted_costs) + 1)
    efficient_quantity = int(benchmark["efficient_quantity"])

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.step(q_values, sorted_values, where="mid", marker="o", label="buyer values")
    ax.step(q_costs, sorted_costs, where="mid", marker="o", label="seller costs")
    ax.axvline(
        efficient_quantity,
        color="black",
        linestyle=":",
        linewidth=1.3,
        label="$Q^{\\ast}$",
    )
    ax.axhspan(
        float(benchmark["price_low"]),
        float(benchmark["price_high"]),
        color="#9ecae1",
        alpha=0.28,
        label="$P^{\\ast}$ band",
    )
    ax.set_xlabel("Unit rank")
    ax.set_ylabel("Induced value or cost")
    ax.set_title("Stepped Demand and Supply in the Baseline Market")
    ax.set_xlim(0.5, max(len(sorted_values), len(sorted_costs)) + 0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def make_transaction_schedule_figure(
    values: np.ndarray,
    costs: np.ndarray,
    trades: pd.DataFrame,
    benchmark: dict[str, object],
) -> plt.Figure:
    """Overlay realized transactions on the induced-value schedule."""
    sorted_values = benchmark["sorted_values"]
    sorted_costs = benchmark["sorted_costs"]
    q_values = np.arange(1, len(sorted_values) + 1)
    q_costs = np.arange(1, len(sorted_costs) + 1)
    efficient_quantity = int(benchmark["efficient_quantity"])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.step(
        q_values,
        sorted_values,
        where="mid",
        marker="o",
        color="#1f77b4",
        linewidth=1.9,
        alpha=0.9,
        label="buyer values",
    )
    ax.step(
        q_costs,
        sorted_costs,
        where="mid",
        marker="o",
        color="#d62728",
        linewidth=1.9,
        alpha=0.9,
        label="seller costs",
    )
    ax.axhspan(
        float(benchmark["price_low"]),
        float(benchmark["price_high"]),
        color="#9ecae1",
        alpha=0.26,
        label="$P^{\\ast}$ band",
    )
    ax.axvline(
        efficient_quantity,
        color="black",
        linestyle=":",
        linewidth=1.3,
        label="$Q^{\\ast}$",
    )

    positive_surplus = trades["Surplus"] > 0.0
    trade_x = trades["Trade"].to_numpy(dtype=float)
    segment_colors = np.where(positive_surplus, "#2ca02c", "#d62728")
    ax.vlines(
        trade_x,
        trades["Seller cost"],
        trades["Buyer value"],
        color=segment_colors,
        alpha=0.22,
        linewidth=3.0,
        zorder=1,
    )
    positive_trades = trades[positive_surplus]
    nonpositive_trades = trades[~positive_surplus]
    if not positive_trades.empty:
        ax.scatter(
            positive_trades["Trade"],
            positive_trades["Price"],
            s=86,
            color="#2ca02c",
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
            label="accepted price: positive surplus",
        )
    if not nonpositive_trades.empty:
        ax.scatter(
            nonpositive_trades["Trade"],
            nonpositive_trades["Price"],
            s=86,
            color="#d62728",
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
            label="accepted price: nonpositive surplus",
        )
    if len(trades) <= 10:
        for _, row in trades.iterrows():
            ax.annotate(
                f"{int(row['Trade'])}",
                (row["Trade"], row["Price"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#222222",
            )

    max_rank = max(len(sorted_values), len(sorted_costs), int(trades["Trade"].max()))
    y_min = min(float(np.min(sorted_costs)), float(trades["Seller cost"].min())) - 6.0
    y_max = max(float(np.max(sorted_values)), float(trades["Buyer value"].max())) + 6.0
    ax.set_xlim(0.5, max_rank + 0.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Unit rank and realized trade order")
    ax.set_ylabel("Price, value, or cost")
    ax.set_title("Transactions on the Demand and Supply Schedule")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def make_market_type_figure(summary: pd.DataFrame) -> plt.Figure:
    """Compare allocative efficiency and price dispersion across market types."""
    labels = summary["Market type"].str.replace(" x ", "\nx ", regex=False)
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))

    axes[0].bar(x, 100.0 * summary["Allocative efficiency"], color="#2ca02c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=0)
    axes[0].set_ylabel("AE (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Surplus Recovery")

    axes[1].bar(x, summary["Price SD"], color="#9467bd")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=0)
    axes[1].set_ylabel("Transaction price SD")
    axes[1].set_title("Price Dispersion")

    fig.tight_layout()
    return fig


def make_agent_mix_figure(
    summary: pd.DataFrame,
    benchmark: dict[str, object],
) -> plt.Figure:
    """Compare ZIP-style adaptation with all-ZIC trading."""
    labels = ["All ZIC", "One ZIP\npair", "All ZIP"]
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))

    axes[0].errorbar(
        x,
        summary["Mean price"],
        yerr=summary["Price SD"],
        fmt="o",
        markersize=7,
        capsize=5,
        color="#1f77b4",
    )
    axes[0].axhspan(
        float(benchmark["price_low"]),
        float(benchmark["price_high"]),
        color="#9ecae1",
        alpha=0.25,
        label="$P^{\\ast}$ band",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Mean price with one SD")
    axes[0].set_title("Price Stability")
    axes[0].legend(loc="best")

    axes[1].bar(x, 100.0 * summary["Allocative efficiency"], color="#2ca02c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("AE (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Allocation")

    fig.tight_layout()
    return fig


def main() -> None:
    setup_style()
    specs = market_specs()
    baseline = specs["balanced"]
    values = np.array(baseline.values, dtype=float)
    costs = np.array(baseline.costs, dtype=float)
    benchmark = competitive_benchmark(values, costs)
    trades, _ = simulate_double_auction(baseline, seed=11)

    if trades.empty:
        raise RuntimeError("Baseline market produced no trades.")

    baseline_metrics = session_metrics(baseline, trades, benchmark)
    market_summary_numeric = run_market_type_sweep(specs)
    agent_mix_numeric = run_agent_mix_sweep(baseline)
    transaction_table = display_transaction_log(trades)
    market_summary = display_market_summary(market_summary_numeric)
    agent_mix_summary = display_agent_mix_summary(agent_mix_numeric)

    Path("figures").mkdir(exist_ok=True)
    Path("tables").mkdir(exist_ok=True)

    fig_ds = make_demand_supply_figure(values, costs, benchmark)
    save_figure(fig_ds, "figures/demand-supply-schedule.png", dpi=150)

    fig_ts = make_transaction_schedule_figure(values, costs, trades, benchmark)
    save_figure(fig_ts, "figures/transaction-schedule.png", dpi=150)

    transaction_table.to_csv("tables/transaction-log.csv", index=False)

    fig_mt = make_market_type_figure(market_summary_numeric)
    save_figure(fig_mt, "figures/market-type-comparison.png", dpi=150)

    market_summary.to_csv("tables/market-type-summary.csv", index=False)

    fig_am = make_agent_mix_figure(agent_mix_numeric, benchmark)
    save_figure(fig_am, "figures/agent-mix-comparison.png", dpi=150)

    agent_mix_summary.to_csv("tables/agent-mix-summary.csv", index=False)

    save_thumbnail("figures/demand-supply-schedule.png", "figures/thumb.png")

    print(
        f"Figures and tables written. "
        f"Baseline allocative efficiency: "
        f"{format_percent(float(baseline_metrics['Allocative efficiency']))}"
    )


if __name__ == "__main__":
    main()
