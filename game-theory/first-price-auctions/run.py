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
        "First-Price Auctions and Bid Shading",
        "Private values, equilibrium bidding, and a direct unilateral-deviation check.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A first-price sealed-bid auction is a pricing problem under private information. "
        "A bidder knows its own value, knows the distribution of rival values, and pays its "
        "own bid when it wins. Lowering the bid raises surplus conditional on winning, but "
        "also lowers the chance of being the highest bidder.\n\n"
        "The uniform independent-private-values case is useful because that tradeoff has an "
        "exact symmetric Bayesian Nash equilibrium. The numerical part of the tutorial is "
        "therefore not a black-box equilibrium search. It is a unilateral-deviation check: "
        "if rivals use the exact bid rule, a grid best response should return the same bid. "
        "That is the Bayesian-game analogue of the no-deviation checks in "
        "[normal-form games](../normal-form-games/), with types replacing payoff-table cells."
    )

    report.add_equations(r"""
There are $n$ risk-neutral bidders. Bidder $i$ has private value
$v_i \sim U[0,1]$, independently across bidders. A pure symmetric strategy is
an increasing bid function $b(v)$.

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

The bid is below value because the winner pays its own bid. If a type $v$
deviates to dollar bid $\hat b$ while opponents use $b^{*}$, the rival value
threshold beaten by $\hat b$ is

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
        "The closed-form bid rule is the economic solution. The grid calculation is a "
        "diagnostic for strategic optimality: it asks whether any type wants to move away "
        "from the proposed bid when all other bidders use the same rule.\n\n"
        "```text\n"
        "Algorithm: first-price bid rule and unilateral-deviation check\n"
        "Inputs: bidder count n, type grid V, bid grid B(v) on [0,v]\n"
        "Outputs: exact bid rule b*(v) and max best-response residual Delta_n\n\n"
        "1. For each type v in V, set the exact bid b*(v) = ((n-1)/n) v.\n"
        "2. For each candidate bid bhat in B(v), compute x(bhat)=min{n bhat/(n-1), 1}.\n"
        "3. Evaluate pi(v,bhat) = (v-bhat) x(bhat)^(n-1).\n"
        "4. Let BR(v) be the bid on the grid with the highest pi(v,bhat).\n"
        "5. Report Delta_n = max_v |BR(v)-b*(v)|.\n"
        "```\n\n"
        "The residual is a no-profitable-deviation diagnostic. A small value means the "
        "finite bid grid is selecting the analytic equilibrium bid, up to grid error."
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
        "The bid functions make the central comparative static visible. With two bidders, "
        "the equilibrium bid is only one half of value. As the number of rivals rises, the "
        "cost of shading increases because a small reduction in the bid gives up more win "
        "probability. The dashed 45-degree line is truthful bidding, not the first-price "
        "equilibrium except in the limit as competition becomes very large."
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
        "payoff curve is single-peaked at the exact bid. The red point is the grid "
        "best response. Its overlap with the analytic vertical line is the concrete "
        "equilibrium check: conditional on rivals using $b^{*}$, this type does not "
        "want to shade more or bid more aggressively."
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
            "The table keeps the analytic equilibrium separate from the grid diagnostic. "
            "Residuals are largest when the exact bid falls between adjacent grid bids."
        ),
    )

    report.add_takeaway(
        "The first-price auction turns private information into bid shading. In the uniform "
        "symmetric benchmark, the equilibrium bid is a constant fraction of value, and that "
        "fraction rises with competition. The grid best-response calculation is useful because "
        "it verifies the economic restriction that defines Bayesian Nash equilibrium: no type "
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
