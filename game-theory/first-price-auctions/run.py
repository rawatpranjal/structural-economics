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
    report = ModelReport(
        "First-Price Auctions, Bid Shading, and Deviation Checks",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "In a first-price sealed-bid auction, each bidder pays its own bid if it wins. "
        "Bidding value helps win, but it leaves no surplus.\n\n"
        "The object is a symmetric bid rule for independent private values. With uniform "
        "values and risk-neutral bidders, equilibrium bids are a constant fraction of "
        "value.\n\n"
        "The computation checks whether that rule is optimal type by type. Given rival "
        "strategies, a bid grid should recover the analytic bid as the best response."
    )

    report.add_equations(r"""
An auction has $n$ risk-neutral bidders. Bidder $i$ observes a private value
$v_i \sim U[0,1]$, independently across bidders, and submits one sealed bid.
The highest bid wins, and the winner pays its own bid. A symmetric Bayesian
Nash strategy is an increasing bid function $b(v)$.

Under uniform values, the equilibrium bid is

$$
b^{*}(v)=\frac{n-1}{n}v.
$$

The rule shades value by $v/n$.

To check optimality, fix a type $v$. Let that bidder choose a dollar bid
$\hat b$ while opponents use $b^{*}$. The rival value threshold beaten by
$\hat b$ is

$$
x(\hat b)=\min\left(\frac{n}{n-1}\hat b,\ 1\right).
$$

The probability of winning is

$$
\Pr(\text{win}\mid \hat b)=x(\hat b)^{n-1},
$$

Expected payoff is

$$
\pi(v,\hat b)=(v-\hat b)x(\hat b)^{n-1}.
$$
""")

    report.add_model_setup(
        "| Object | Value | Role |\n"
        "|---|---:|---|\n"
        "| Value distribution | $U[0,1]$ | Independent private values |\n"
        "| Risk preferences | Risk neutral | Payoff is value minus payment when winning |\n"
        "| Bidder counts | 2, 3, 5, 10 | More or fewer rivals |\n"
        f"| Deviation grid | 2,001 bids per value | Best-response check for each $v$ |\n"
        f"| Check values | {len(check_values)} values in [0.05, 0.95] | Types used for residuals |\n"
        f"| Focal deviation plot | $n={focal_n}$, $v={focal_value:.1f}$ | Payoff shape for one bidder type |"
    )

    report.add_solution_method(
        "The code treats the formula as a candidate equilibrium. It holds opponent "
        "behavior fixed and computes a grid best response for each checked type.\n\n"
        "```text\n"
        "Algorithm: bid shading and unilateral-deviation check\n"
        "Inputs: bidder count n, type grid V, bid grid B(v) on [0,v]\n"
        "Outputs: exact bid rule b*(v) and max best-response residual Delta_n\n\n"
        "1. For each type v in V, set the exact bid b*(v) = ((n-1)/n) v.\n"
        "2. For each candidate bid bhat in B(v), compute x(bhat)=min{n bhat/(n-1), 1}.\n"
        "3. Evaluate pi(v,bhat) = (v-bhat) x(bhat)^(n-1).\n"
        "4. Let BR(v) be the bid on the grid with the highest pi(v,bhat).\n"
        "5. Report Delta_n = max_v |BR(v)-b*(v)|.\n"
        "```\n\n"
        "Each residual is the distance between the grid best response and the analytic "
        "bid. Small residuals mean the candidate passes the no-deviation check."
    )

    fig, ax = plt.subplots()
    for n_bidders in bidder_counts:
        ax.plot(values, equilibrium_bid(values, n_bidders), label=f"{n_bidders} bidders")
    ax.plot(values, values, color="black", linestyle="--", linewidth=1.2, label="Truthful value")
    ax.set_xlabel("Value $v$")
    ax.set_ylabel("Bid $b(v)$")
    ax.set_title("Equilibrium Bidding Under Uniform Private Values")
    ax.legend()
    report.add_results(
        "The bid functions show less shading when more bidders compete. With two bidders, "
        "a bidder bids one half of value. With ten bidders, the bid is close to value. "
        "Extra rivals make low bids more costly because they reduce win probability."
    )
    report.add_figure(
        "figures/bid-functions.png",
        "Equilibrium first-price bid functions by bidder count",
        fig,
    )

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
    report.add_results(
        f"For value {focal_value:.1f} with {focal_n - 1} rivals, the payoff curve peaks "
        "at the analytic bid. Lower bids raise margin only when they still win. Higher "
        "bids buy win probability at a higher payment. The grid best response sits on "
        "the analytic bid."
    )
    report.add_figure(
        "figures/best-response-check.png",
        "Grid best response compared with the exact equilibrium bid",
        fig2,
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
        description=(
            "Residuals measure grid error. Some exact bids lie between grid points."
        ),
    )

    report.add_takeaway(
        "First-price auctions reward bid shading. In the uniform symmetric benchmark, "
        "the equilibrium bid is $((n-1)/n)v$.\n\n"
        "The useful computational check is type by type. Hold rival strategies fixed "
        "and verify that no bid on the grid improves payoff."
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
