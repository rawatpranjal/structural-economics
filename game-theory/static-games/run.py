#!/usr/bin/env python3
"""Cournot oligopoly and best-response dynamics."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


def cournot_best_response(
    q_other: float | np.ndarray,
    a: float,
    b: float,
    c: float,
) -> float | np.ndarray:
    """Firm i's best response in a linear Cournot duopoly."""
    return np.maximum(0.0, (a - c - b * q_other) / (2.0 * b))


def iterate_best_responses(
    q0: tuple[float, float],
    a: float,
    b: float,
    c: float,
    steps: int,
    damping: float,
) -> np.ndarray:
    """Iterate damped simultaneous Cournot best responses."""
    path = np.zeros((steps + 1, 2))
    path[0] = np.asarray(q0, dtype=float)

    for t in range(steps):
        q1, q2 = path[t]
        target = np.array(
            [
                cournot_best_response(q2, a, b, c),
                cournot_best_response(q1, a, b, c),
            ]
        )
        path[t + 1] = (1.0 - damping) * path[t] + damping * target

    return path


def fixed_point_residual(q: np.ndarray, a: float, b: float, c: float) -> float:
    """Maximum unilateral best-response error at a candidate quantity pair."""
    br = np.array(
        [
            cournot_best_response(q[1], a, b, c),
            cournot_best_response(q[0], a, b, c),
        ]
    )
    return float(np.max(np.abs(q - br)))


def price(total_quantity: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    """Inverse demand."""
    return a - b * total_quantity


def profit(
    q_i: float | np.ndarray,
    q_j: float | np.ndarray,
    a: float,
    b: float,
    c: float,
) -> float | np.ndarray:
    """Firm i profit in the Cournot game."""
    return (price(q_i + q_j, a, b) - c) * q_i


def surplus_table(a: float, b: float, c: float, q_nash: float) -> pd.DataFrame:
    """Compare monopoly, duopoly, and competitive output levels."""
    q_monopoly = (a - c) / (2.0 * b)
    q_competitive = (a - c) / b
    rows = []

    for name, total_output, firms in [
        ("Monopoly", q_monopoly, 1),
        ("Cournot duopoly", 2.0 * q_nash, 2),
        ("Perfect competition", q_competitive, np.inf),
    ]:
        market_price = price(total_output, a, b)
        if np.isinf(firms):
            firm_profit = 0.0
        elif firms == 1:
            firm_profit = (market_price - c) * total_output
        else:
            firm_profit = profit(q_nash, q_nash, a, b, c)

        rows.append(
            {
                "Market structure": name,
                "Total output": f"{total_output:.3f}",
                "Price": f"{market_price:.3f}",
                "Profit per firm": f"{firm_profit:.3f}",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    a = 10.0
    b = 1.0
    c = 2.0
    damping = 0.65
    steps = 32
    starts = [(0.5, 7.0), (7.0, 0.5), (7.0, 7.0)]

    q_star = (a - c) / (3.0 * b)
    p_star = price(2.0 * q_star, a, b)
    pi_star = profit(q_star, q_star, a, b, c)
    q_monopoly = (a - c) / (2.0 * b)
    q_competitive = (a - c) / b
    paths = {
        q0: iterate_best_responses(q0, a, b, c, steps=steps, damping=damping)
        for q0 in starts
    }

    setup_style()
    report = ModelReport(
        "Cournot Oligopoly and Best-Response Dynamics",
        "A static quantity game solved by closed-form Nash conditions and checked by best-response iteration.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Cournot competition is a small static game with a large economic lesson: "
        "market power comes from each firm's recognition that its own output moves "
        "the market price. The Nash quantity is not the joint-profit maximum and "
        "not the competitive quantity. It is the point where each firm is already "
        "choosing its optimal quantity given the other firm's quantity.\n\n"
        "Two solution views run side by side. The first is the closed-form first-order "
        "condition. The second iterates best responses and reports a fixed-point residual. "
        "The iteration matters because larger games usually do not have a one-line "
        "equilibrium formula."
    )

    report.add_equations(
        r"""
Two firms choose quantities $q_1$ and $q_2$ simultaneously. Total output is
$Q=q_1+q_2$, inverse demand is

$$
P(Q)=a-bQ,
$$

and firm $i$ has constant marginal cost $c$. Given $q_j$, firm $i$ solves

$$
\max_{q_i \geq 0}\ (a-b(q_i+q_j)-c)q_i.
$$

The interior first-order condition gives the best response

$$
BR_i(q_j)=\frac{a-c-bq_j}{2b}.
$$

A symmetric Nash equilibrium satisfies $q_i=q_j=q^{*}$ and
$q^{*}=BR_i(q^{*})$, so

$$
q^{*}=\frac{a-c}{3b},\qquad
P^{*}=a-2bq^{*}.
$$

The comparison points are also useful:

$$
Q^{M}=\frac{a-c}{2b},\qquad
Q^{C}=\frac{a-c}{b},
$$

where $Q^{M}$ is monopoly output and $Q^{C}$ is the competitive output at
price equal to marginal cost.
"""
    )

    report.add_model_setup(
        "| Object | Value | Meaning |\n"
        "|---|---:|---|\n"
        f"| $a$ | {a:.1f} | Demand intercept |\n"
        f"| $b$ | {b:.1f} | Demand slope |\n"
        f"| $c$ | {c:.1f} | Marginal cost |\n"
        f"| $q^{{*}}$ | {q_star:.3f} | Nash output per firm |\n"
        f"| $P^{{*}}$ | {p_star:.3f} | Nash price |\n"
        f"| $\\pi^{{*}}$ | {pi_star:.3f} | Nash profit per firm |\n"
        f"| Damping $\\lambda$ | {damping:.2f} | Weight on each new best response |"
    )

    report.add_solution_method(
        "The closed-form solution solves the two first-order conditions directly. "
        "The best-response iteration treats the same equilibrium as a fixed point "
        "of the map $BR(q_1,q_2)=(BR_1(q_2),BR_2(q_1))$.\n\n"
        "```text\n"
        "Algorithm: damped Cournot best-response iteration\n"
        "Inputs: demand parameters a, b, marginal cost c, start q_0, damping lambda\n"
        "Output: quantity path q_t and fixed-point residuals\n\n"
        "1. Start from a candidate pair q_t = (q_{1t}, q_{2t}).\n"
        "2. Compute each firm's best response to the other firm's current output.\n"
        "3. Update q_{t+1} = (1-lambda) q_t + lambda BR(q_t).\n"
        "4. Repeat until max_i |q_{it} - BR_i(q_{-i,t})| is near zero.\n"
        "5. Compare the numerical fixed point with q* = (a-c)/(3b).\n"
        "```\n\n"
        "A small residual matters more than a visually stable path: Nash equilibrium "
        "is a no-profitable-deviation condition, not just convergence of a line on a plot."
    )

    q_grid = np.linspace(0.0, 8.0, 240)
    br = cournot_best_response(q_grid, a, b, c)

    fig1, ax1 = plt.subplots(figsize=(7, 6))
    ax1.plot(q_grid, br, linewidth=2.3, label="$BR_1(q_2)$")
    ax1.plot(br, q_grid, linewidth=2.3, label="$BR_2(q_1)$")
    ax1.scatter(q_star, q_star, color="black", s=60, zorder=5, label=f"Nash q={q_star:.2f}")
    ax1.scatter(
        q_monopoly / 2.0,
        q_monopoly / 2.0,
        color="#4b8f29",
        marker="s",
        s=55,
        zorder=5,
        label="Joint monopoly split",
    )
    for q0, path in paths.items():
        ax1.plot(path[:, 1], path[:, 0], marker="o", markersize=2.5, alpha=0.65, label=f"Start {q0}")
    ax1.set_xlabel("$q_2$")
    ax1.set_ylabel("$q_1$")
    ax1.set_title("Cournot Best Responses and Iteration Paths")
    ax1.set_xlim(0.0, 8.0)
    ax1.set_ylim(0.0, 8.0)
    ax1.legend(fontsize=8)

    report.add_results(
        "The best-response curves cross at the Nash quantity. The joint-monopoly "
        "split is inside the diagram but not an equilibrium: each firm would rather "
        "expand output if the rival stayed at the collusive quantity. The damped "
        "paths show how the same Nash condition can be reached by iteration from "
        "different starting points."
    )
    report.add_figure(
        "figures/cournot-best-response.png",
        "Cournot best-response curves and damped iteration paths",
        fig1,
    )

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for q0, path in paths.items():
        residuals = [fixed_point_residual(path[t], a, b, c) for t in range(len(path))]
        ax2.semilogy(residuals, marker="o", markersize=3, label=f"Start {q0}")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Fixed-point residual")
    ax2.set_title("Best-Response Residual")
    ax2.legend()

    report.add_results(
        "The residual falls quickly because this linear duopoly has a stable "
        "best-response map after damping. Reporting it ties the numerical exercise to "
        "the economic definition of equilibrium."
    )
    report.add_figure(
        "figures/residuals.png",
        "Fixed-point residuals for damped best-response iteration",
        fig2,
    )

    total_output = np.linspace(0.0, a / b, 240)
    market_price = price(total_output, a, b)
    consumer_surplus = 0.5 * b * total_output**2
    producer_surplus = (market_price - c) * total_output
    total_surplus = consumer_surplus + producer_surplus

    fig3, ax3 = plt.subplots(figsize=(7, 4.8))
    ax3.plot(total_output, consumer_surplus, linewidth=2.0, label="Consumer surplus")
    ax3.plot(total_output, producer_surplus, linewidth=2.0, label="Producer surplus")
    ax3.plot(total_output, total_surplus, color="black", linestyle="--", linewidth=1.8, label="Total surplus")
    ax3.axvline(q_monopoly, color="#b85c00", linestyle=":", label=f"Monopoly Q={q_monopoly:.1f}")
    ax3.axvline(2.0 * q_star, color="#1b6ca8", linestyle=":", label=f"Cournot Q={2*q_star:.1f}")
    ax3.axvline(q_competitive, color="#4b8f29", linestyle=":", label=f"Competitive Q={q_competitive:.1f}")
    ax3.set_xlabel("Total output $Q$")
    ax3.set_ylabel("Surplus")
    ax3.set_title("Output and Surplus Benchmarks")
    ax3.legend(fontsize=8)

    report.add_results(
        "The welfare comparison is the economic reason the equilibrium matters. "
        "Cournot output lies between monopoly and perfect competition, so the price "
        "is also intermediate. The exact numbers are calibration-specific, but the "
        "ranking is the standard Cournot logic."
    )
    report.add_figure(
        "figures/welfare-analysis.png",
        "Monopoly, Cournot, and competitive output benchmarks",
        fig3,
    )

    rows = []
    for q0, path in paths.items():
        q_final = path[-1]
        rows.append(
            {
                "Initial q": str(q0),
                "Final q1": f"{q_final[0]:.4f}",
                "Final q2": f"{q_final[1]:.4f}",
                "Residual": f"{fixed_point_residual(q_final, a, b, c):.2e}",
            }
        )
    rows.append(
        {
            "Initial q": "Closed form",
            "Final q1": f"{q_star:.4f}",
            "Final q2": f"{q_star:.4f}",
            "Residual": f"{fixed_point_residual(np.array([q_star, q_star]), a, b, c):.2e}",
        }
    )

    report.add_table(
        "tables/convergence-summary.csv",
        "Best-Response Convergence",
        pd.DataFrame(rows),
    )
    report.add_table(
        "tables/cournot-comparison.csv",
        "Cournot Benchmarks",
        surplus_table(a, b, c, q_star),
    )

    report.add_takeaway(
        "Cournot equilibrium is a fixed point with economic content: each firm is "
        "already choosing its profit-maximizing quantity given the rival's output. "
        "Closed form makes that condition transparent here. Best-response iteration "
        "is the computational version of the same idea, and the residual verifies "
        "that the iteration has actually reached a Nash equilibrium."
    )

    report.add_references(
        [
            "Cournot, A. A. (1838/1897). *Researches into the Mathematical Principles of the Theory of Wealth*. English translation.",
            "Fudenberg, D. and Tirole, J. (1991). *Game Theory*. MIT Press.",
            "Tirole, J. (1988). *The Theory of Industrial Organization*. MIT Press, Ch. 5.",
            "Vives, X. (1999). *Oligopoly Pricing: Old Ideas and New Tools*. MIT Press.",
        ]
    )

    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
