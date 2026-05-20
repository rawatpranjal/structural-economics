#!/usr/bin/env python3
"""First-price auctions with symmetric independent private values.

The tutorial uses the exact uniform-IPV Bayesian Nash bid function and treats
grid best responses as a unilateral-deviation check.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


def equilibrium_bid(value: np.ndarray | float, n_bidders: int) -> np.ndarray | float:
    """Symmetric first-price auction bid for U[0,1] private values."""
    return (n_bidders - 1.0) / n_bidders * value


def win_probability(bid: np.ndarray | float, n_bidders: int) -> np.ndarray | float:
    """Probability of winning when opponents use the symmetric equilibrium strategy."""
    if n_bidders < 2:
        raise ValueError("An auction needs at least two bidders.")
    threshold_value = np.clip(n_bidders / (n_bidders - 1.0) * bid, 0.0, 1.0)
    return threshold_value ** (n_bidders - 1)


def expected_payoff(value: float, bid: np.ndarray, n_bidders: int) -> np.ndarray:
    """Expected payoff from bidding `bid` when value is fixed."""
    return np.maximum(value - bid, 0.0) * win_probability(bid, n_bidders)


def grid_best_response(value: float, n_bidders: int, n_grid: int = 2001) -> tuple[float, float]:
    """Compute a grid best response against equilibrium opponents."""
    bids = np.linspace(0.0, value, n_grid)
    payoffs = expected_payoff(value, bids, n_bidders)
    idx = int(np.argmax(payoffs))
    return float(bids[idx]), float(payoffs[idx])


def main() -> None:
    bidder_counts = [2, 3, 5, 10]
    values = np.linspace(0.0, 1.0, 200)
    check_values = np.linspace(0.05, 0.95, 19)
    focal_value = 0.8
    focal_n = 3

    rows = []
    residual_rows = []
    for n_bidders in bidder_counts:
        shading = 1.0 / n_bidders
        max_error = 0.0
        for value in check_values:
            br_bid, _ = grid_best_response(float(value), n_bidders)
            eq_bid = equilibrium_bid(float(value), n_bidders)
            max_error = max(max_error, abs(br_bid - eq_bid))

        rows.append({
            "Bidders": n_bidders,
            "Equilibrium bid rule": f"b*(v)={(n_bidders - 1)}/{n_bidders} v",
            "Shading at v=1": f"{shading:.3f}",
        })
        residual_rows.append({
            "Bidders": n_bidders,
            "Max absolute BR error": f"{max_error:.3e}",
        })

    setup_style()

    fig, ax = plt.subplots()
    for n_bidders in bidder_counts:
        ax.plot(values, equilibrium_bid(values, n_bidders), label=f"{n_bidders} bidders")
    ax.plot(values, values, color="black", linestyle="--", linewidth=1.2, label="Truthful value")
    ax.set_xlabel("Value $v$")
    ax.set_ylabel("Bid $b(v)$")
    ax.set_title("Equilibrium Bidding Under Uniform Private Values")
    ax.legend()
    save_figure(fig, "figures/bid-functions.png", dpi=150)

    bids = np.linspace(0.0, focal_value, 300)
    payoffs = expected_payoff(focal_value, bids, focal_n)
    br_bid, br_payoff = grid_best_response(focal_value, focal_n)
    eq_bid = equilibrium_bid(focal_value, focal_n)

    fig2, ax2 = plt.subplots()
    ax2.plot(bids, payoffs, label=f"Expected payoff, v={focal_value:.1f}")
    ax2.axvline(eq_bid, color="black", linestyle=":", label=f"Exact bid {eq_bid:.3f}")
    ax2.scatter(br_bid, br_payoff, color="crimson", zorder=5, label=f"Grid BR {br_bid:.3f}")
    ax2.set_xlabel("Bid")
    ax2.set_ylabel("Expected payoff")
    ax2.set_title("Unilateral-Deviation Payoff Against Equilibrium Rivals")
    ax2.legend()
    save_figure(fig2, "figures/best-response-check.png", dpi=150)

    Path("tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv("tables/auction-summary.csv", index=False)
    pd.DataFrame(residual_rows).to_csv("tables/best-response-residuals.csv", index=False)

    save_thumbnail("figures/bid-functions.png", "figures/thumb.png")
    print(f"Done: 2 figures + 2 tables")


if __name__ == "__main__":
    main()
