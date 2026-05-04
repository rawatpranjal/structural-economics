#!/usr/bin/env python3
"""First-price auctions with symmetric independent private values.

Uses the closed-form equilibrium for uniform values and verifies it with a
direct grid best-response check.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


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


def expected_revenue(n_bidders: int) -> float:
    """Expected winning bid under the symmetric U[0,1] equilibrium."""
    return (n_bidders - 1.0) / (n_bidders + 1.0)


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
            "Bid function": f"b(v)={(n_bidders - 1)}/{n_bidders} v",
            "Bid shading at v=1": f"{shading:.3f}",
            "Expected revenue": f"{expected_revenue(n_bidders):.3f}",
        })
        residual_rows.append({
            "Bidders": n_bidders,
            "Max grid BR error": f"{max_error:.3e}",
        })

    setup_style()
    report = ModelReport(
        "First-Price Auctions and Bid Shading",
        "Bayesian Nash equilibrium in a symmetric independent private values auction.",
    )

    report.add_overview(
        "A first-price sealed-bid auction is a Bayesian game: each bidder knows their own value "
        "but not rivals' values. In the symmetric independent private values model with uniform "
        "values, the equilibrium is a simple bid-shading rule. The code uses the closed-form "
        "solution and then checks it by direct best-response search on a grid."
    )

    report.add_equations(r"""
There are $n$ risk-neutral bidders. Values are independently drawn from $U[0,1]$.
In a symmetric increasing equilibrium, bidder $i$ with value $v$ bids
$$
b(v) = \frac{n-1}{n}v.
$$

If a bidder with value $v$ deviates to bid $b$ while opponents use the equilibrium strategy,
the probability of winning is
$$
\Pr(\text{win} \mid b) =
\left(\frac{n}{n-1}b\right)^{n-1},
$$
for bids below the highest equilibrium bid. Expected payoff is
$$
\pi(v,b) = (v-b)\Pr(\text{win} \mid b).
$$
""")

    report.add_model_setup(
        "The baseline assumes independent $U[0,1]$ values, risk-neutral bidders, no reserve price, "
        "and no binding ties. The script compares auctions with 2, 3, 5, and 10 bidders."
    )

    report.add_solution_method(
        "**Closed form:** use the symmetric equilibrium bid function "
        "$b(v)=\\frac{n-1}{n}v$.\n\n"
        "**Numerical check:** for a grid of values, search over bids and verify that the best "
        "response against equilibrium opponents is close to the closed-form bid."
    )

    fig, ax = plt.subplots()
    for n_bidders in bidder_counts:
        ax.plot(values, equilibrium_bid(values, n_bidders), label=f"n={n_bidders}")
    ax.plot(values, values, color="black", linestyle="--", linewidth=1, label="Truthful bid")
    ax.set_xlabel("Value $v$")
    ax.set_ylabel("Bid $b(v)$")
    ax.set_title("Equilibrium Bid Shading")
    ax.legend()
    report.add_figure(
        "figures/bid-functions.png",
        "First-price bidders shade bids below values",
        fig,
        description=(
            "More competition reduces bid shading. With many rivals, winning becomes more valuable "
            "at the margin, so bids move closer to values."
        ),
    )

    bids = np.linspace(0.0, focal_value, 300)
    payoffs = expected_payoff(focal_value, bids, focal_n)
    br_bid, br_payoff = grid_best_response(focal_value, focal_n)
    eq_bid = equilibrium_bid(focal_value, focal_n)

    fig2, ax2 = plt.subplots()
    ax2.plot(bids, payoffs, label=f"Expected payoff, v={focal_value:.1f}")
    ax2.axvline(eq_bid, color="black", linestyle=":", label=f"Equilibrium bid {eq_bid:.3f}")
    ax2.scatter(br_bid, br_payoff, color="crimson", zorder=5, label="Grid best response")
    ax2.set_xlabel("Bid")
    ax2.set_ylabel("Expected payoff")
    ax2.set_title("Best-Response Check Against Equilibrium Opponents")
    ax2.legend()
    report.add_figure(
        "figures/best-response-check.png",
        "Closed-form bid matches the grid best response",
        fig2,
        description=(
            "For a bidder with value 0.8 in a 3-bidder auction, the payoff-maximizing deviation "
            "is the equilibrium bid."
        ),
    )

    revenue = [expected_revenue(n) for n in range(2, 21)]
    fig3, ax3 = plt.subplots()
    ax3.plot(range(2, 21), revenue, marker="o")
    ax3.set_xlabel("Number of bidders")
    ax3.set_ylabel("Expected winning bid")
    ax3.set_title("Competition Raises Expected Revenue")
    report.add_figure(
        "figures/revenue-by-bidders.png",
        "Expected revenue rises with the number of bidders",
        fig3,
    )

    report.add_table(
        "tables/auction-summary.csv",
        "Auction Summary",
        pd.DataFrame(rows),
    )
    report.add_table(
        "tables/best-response-residuals.csv",
        "Best-Response Check",
        pd.DataFrame(residual_rows),
        description="The residual is the largest absolute difference between the closed-form bid and a grid best response.",
    )

    report.add_takeaway(
        "The symmetric auction model is a compact example of Bayesian Nash equilibrium. A bidder "
        "trades off a lower payment conditional on winning against a lower probability of winning. "
        "The equilibrium bid rule solves that tradeoff, and the grid check verifies the strategic "
        "optimality without introducing a specialized auction package."
    )

    report.add_references([
        "[Vickrey, W. (1961). Counterspeculation, Auctions, and Competitive Sealed Tenders. *Journal of Finance*, 16(1), 8-37.](https://doi.org/10.1111/j.1540-6261.1961.tb02789.x)",
        "[Riley, J. G. and Samuelson, W. F. (1981). Optimal Auctions. *American Economic Review*, 71(3), 381-392.](https://www.jstor.org/stable/1802786)",
        "[Krishna, V. (2009). *Auction Theory*, 2nd ed. Academic Press.](https://shop.elsevier.com/books/auction-theory/krishna/978-0-12-374507-1)",
    ])

    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
