#!/usr/bin/env python3
"""First-price auctions with symmetric independent private values.

The tutorial uses the exact uniform-IPV Bayesian Nash bid function and treats
grid best responses as a unilateral-deviation audit.
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
            "Equilibrium bid rule": f"b*(v)={(n_bidders - 1)}/{n_bidders} v",
            "Shading at v=1": f"{shading:.3f}",
            "Expected revenue": f"{expected_revenue(n_bidders):.3f}",
        })
        residual_rows.append({
            "Bidders": n_bidders,
            "Max absolute BR error": f"{max_error:.3e}",
        })

    setup_style()
    report = ModelReport(
        "First-Price Auctions, Bid Shading, and Deviation Checks",
        "Private-value bidding, the symmetric equilibrium rule, and a direct grid audit.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A first-price sealed-bid auction forces each bidder to price its private "
        "information. Suppose a bidder values the object at 0.80. A bid close to 0.80 "
        "wins often but leaves little surplus. A lower bid gives a larger margin if it "
        "wins, but it loses more often to rival bids. Equilibrium bidding balances those "
        "two forces for every possible value, not only for this one bidder.\n\n"
        "The uniform independent-private-values model lets us write the symmetric "
        "Bayesian Nash bid rule in closed form. The computation then has a clear job: "
        "take that candidate rule and check it type by type. If rivals use the analytic "
        "rule, a grid best response over possible bids should return the same bid, up to "
        "grid error. This is the Bayesian-game version of the no-deviation checks in "
        "[normal-form games](../normal-form-games/), with private values replacing "
        "payoff-table cells."
    )

    report.add_equations(r"""
An auction has $n$ risk-neutral bidders. Bidder $i$ observes a private value
$v_i \sim U[0,1]$, independently across bidders, and submits one sealed bid.
The winner pays its own bid. A pure symmetric strategy is an increasing bid
function $b(v)$.

For a general distribution $F$, the symmetric first-price bid rule satisfies

$$
b(v)
= v-\frac{\int_0^v F(t)^{n-1}\,dt}{F(v)^{n-1}},
\qquad v>0,
$$

with $b(0)=0$. With $F(v)=v$ on $[0,1]$, this becomes

$$
b^{*}(v)=\frac{n-1}{n}v.
$$

The bid is below value because the winner pays its own bid. To check the
equilibrium restriction, fix a type $v$ and let that bidder deviate to dollar
bid $\hat b$ while opponents use $b^{*}$. The rival value threshold beaten by
$\hat b$ is

$$
x(\hat b)=\min\left(\frac{n}{n-1}\hat b,\ 1\right).
$$

The probability of winning is

$$
\Pr(\text{win}\mid \hat b)=x(\hat b)^{n-1},
$$

and expected payoff is

$$
\pi(v,\hat b)=(v-\hat b)x(\hat b)^{n-1}.
$$

Expected revenue under the equilibrium is

$$
R_n = E[b^{*}(V_{n:n})]
    = \frac{n-1}{n}E[V_{n:n}]
    = \frac{n-1}{n+1},
$$

which is also $E[V_{n-1:n}]$, the expected second-highest value in the uniform
auction.
""")

    report.add_model_setup(
        "| Object | Value | Role |\n"
        "|---|---:|---|\n"
        "| Value distribution | $U[0,1]$ | Independent private values |\n"
        "| Risk preferences | Risk neutral | Payoff is value minus payment when winning |\n"
        "| Reserve price | None | Every nonnegative bid is admissible |\n"
        "| Bidder counts | 2, 3, 5, 10 | Comparative statics in competition |\n"
        f"| Deviation grid | 2,001 bids per value | Best-response check for each $v$ |\n"
        f"| Check values | {len(check_values)} values in [0.05, 0.95] | Types used for residuals |\n"
        f"| Focal deviation plot | $n={focal_n}$, $v={focal_value:.1f}$ | Payoff shape for one bidder type |"
    )

    report.add_solution_method(
        "The analytic bid function gives the equilibrium benchmark. The numerical work "
        "asks whether that benchmark satisfies the type-by-type optimality condition. For "
        "each bidder count, the code evaluates the closed-form rule, searches a fine grid "
        "of deviations for each checked value, and records the largest distance between "
        "the grid best response and the analytic bid.\n\n"
        "```text\n"
        "Algorithm: bid shading and unilateral-deviation audit\n"
        "Inputs: bidder count n, type grid V, bid grid B(v) on [0,v]\n"
        "Outputs: exact bid rule b*(v) and max best-response residual Delta_n\n\n"
        "1. For each type v in V, set the exact bid b*(v) = ((n-1)/n) v.\n"
        "2. For each candidate bid bhat in B(v), compute x(bhat)=min{n bhat/(n-1), 1}.\n"
        "3. Evaluate pi(v,bhat) = (v-bhat) x(bhat)^(n-1).\n"
        "4. Let BR(v) be the bid on the grid with the highest pi(v,bhat).\n"
        "5. Report Delta_n = max_v |BR(v)-b*(v)|.\n"
        "```\n\n"
        "This audit approximates the best response against the candidate strategy. It does "
        "not estimate a new equilibrium. A small residual means the finite bid grid selects "
        "the analytic bid that the Bayesian Nash equilibrium prescribes."
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
        "The bid functions show how competition disciplines shading. With two bidders, "
        "the equilibrium bid is one half of value. As the number of rivals rises, a small "
        "bid reduction gives up more win probability, so the bidder shades less. The dashed "
        "45-degree line is truthful bidding. First-price bidders stay below it because the "
        "payment equals their own bid."
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
        f"For a bidder with value {focal_value:.1f} facing {focal_n - 1} rivals, the "
        "payoff curve is single-peaked at the exact bid. Bidding lower raises surplus only "
        "when the bidder still wins, while bidding higher buys extra win probability at a "
        "higher payment. The red point is the grid best response, and its overlap with the "
        "analytic vertical line is the equilibrium check for this type."
    )
    report.add_figure(
        "figures/best-response-check.png",
        "Grid best response compared with the exact equilibrium bid",
        fig2,
    )

    revenue_n = np.arange(2, 21)
    revenue = [expected_revenue(n) for n in revenue_n]
    second_price_benchmark = [(n - 1.0) / (n + 1.0) for n in revenue_n]
    fig3, ax3 = plt.subplots()
    ax3.plot(
        revenue_n,
        second_price_benchmark,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="Second-highest value benchmark",
    )
    ax3.scatter(revenue_n, revenue, color="#2b6cb0", label="First-price expected winning bid")
    ax3.set_xlabel("Number of bidders")
    ax3.set_ylabel("Expected winning bid")
    ax3.set_title("Expected Revenue and the Order-Statistic Benchmark")
    ax3.legend()
    report.add_results(
        "The revenue curve is not a simulation artifact. In the uniform model, the expected "
        "first-price winning bid equals the expected second-highest value. The points and "
        "dashed benchmark coincide, which is the revenue-equivalence result in this simple "
        "risk-neutral environment."
    )
    report.add_figure(
        "figures/revenue-by-bidders.png",
        "Expected first-price revenue and second-highest-value benchmark",
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
        description=(
            "The table keeps the model implication separate from the grid diagnostic. "
            "Residuals are largest when the exact bid falls between adjacent grid bids."
        ),
    )

    report.add_takeaway(
        "The first-price auction turns private information into bid shading. In the uniform "
        "symmetric benchmark, the equilibrium bid is a constant fraction of value, and that "
        "fraction rises with competition. The reusable computational move is to express a "
        "candidate equilibrium as a type-indexed payoff comparison, then check that no type "
        "has a profitable unilateral deviation given the strategy used by rival types."
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
