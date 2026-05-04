#!/usr/bin/env python3
"""Best-response dynamics for a Cournot game.

Solves a simple game by iterating best responses and reports fixed-point
residuals. The implementation is deliberately low-code and dependency-light.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


def cournot_best_response(q_other: float | np.ndarray, a: float, b: float, c: float) -> float | np.ndarray:
    """Best response in a linear Cournot duopoly with inverse demand P=a-bQ."""
    return np.maximum(0.0, (a - c - b * q_other) / (2.0 * b))


def iterate_best_responses(
    q0: tuple[float, float],
    a: float,
    b: float,
    c: float,
    steps: int,
    damping: float = 1.0,
) -> np.ndarray:
    """Iterate simultaneous best responses from an initial output pair."""
    path = np.zeros((steps + 1, 2))
    path[0] = np.array(q0, dtype=float)
    for t in range(steps):
        current = path[t]
        target = np.array([
            cournot_best_response(current[1], a, b, c),
            cournot_best_response(current[0], a, b, c),
        ])
        path[t + 1] = (1.0 - damping) * current + damping * target
    return path


def fixed_point_residual(q: np.ndarray, a: float, b: float, c: float) -> float:
    """Maximum best-response residual for a candidate output pair."""
    br = np.array([
        cournot_best_response(q[1], a, b, c),
        cournot_best_response(q[0], a, b, c),
    ])
    return float(np.max(np.abs(q - br)))


def profit(q_i: np.ndarray | float, q_j: np.ndarray | float, a: float, b: float, c: float) -> np.ndarray | float:
    """Cournot profit for firm i."""
    price = a - b * (q_i + q_j)
    return (price - c) * q_i


def main() -> None:
    a = 10.0
    b = 1.0
    c = 2.0
    steps = 32
    initial_conditions = [(0.5, 7.0), (7.0, 0.5), (7.0, 7.0)]
    damping = 0.65

    q_star = (a - c) / (3.0 * b)
    price_star = a - b * (2.0 * q_star)
    profit_star = profit(q_star, q_star, a, b, c)

    paths = {
        q0: iterate_best_responses(q0, a, b, c, steps=steps, damping=damping)
        for q0 in initial_conditions
    }

    setup_style()
    report = ModelReport(
        "Cournot Best-Response Dynamics",
        "Solving a Cournot game by iterating best responses.",
    )

    report.add_overview(
        "Best-response dynamics solve a game by repeatedly asking each player: given what the "
        "other player is doing now, what is my optimal action? When the best-response map is a "
        "contraction, this iteration converges to a Nash equilibrium. When it is not, the same "
        "idea may cycle or require damping."
    )

    report.add_equations(r"""
Two firms choose quantities $q_1$ and $q_2$. Inverse demand is
$$
P(Q) = a - bQ, \qquad Q = q_1 + q_2,
$$
and each firm has constant marginal cost $c$. Firm $i$ solves
$$
\max_{q_i \geq 0} (a - b(q_i + q_j) - c)q_i.
$$

The best response is
$$
BR_i(q_j) = \max\left\{0, \frac{a-c-bq_j}{2b}\right\}.
$$

The symmetric Nash equilibrium solves $q^* = BR(q^*)$:
$$
q^* = \frac{a-c}{3b}.
$$
""")

    report.add_model_setup(
        "| Parameter | Value | Description |\n"
        "|-----------|-------|-------------|\n"
        f"| $a$ | {a:.1f} | Demand intercept |\n"
        f"| $b$ | {b:.1f} | Demand slope |\n"
        f"| $c$ | {c:.1f} | Marginal cost |\n"
        f"| Damping | {damping:.2f} | Weight on the new best response |\n"
        f"| Iterations | {steps} | Number of best-response updates |"
    )

    report.add_solution_method(
        "Start from several initial quantity pairs. At each iteration, compute both firms' "
        "best responses to the previous quantities and update with damping:\n\n"
        "$$q^{t+1} = (1-\\lambda)q^t + \\lambda BR(q^t).$$\n\n"
        "The diagnostic is the fixed-point residual "
        "$\\max_i |q_i - BR_i(q_{-i})|$."
    )

    q_grid = np.linspace(0, 8, 200)
    br = cournot_best_response(q_grid, a, b, c)
    fig, ax = plt.subplots()
    ax.plot(q_grid, br, label="$BR_1(q_2)$")
    ax.plot(br, q_grid, label="$BR_2(q_1)$")
    ax.scatter(q_star, q_star, color="black", s=60, zorder=5, label=f"Nash: {q_star:.2f}")
    for q0, path in paths.items():
        ax.plot(path[:, 1], path[:, 0], marker="o", markersize=3, alpha=0.75, label=f"Path from {q0}")
    ax.set_xlabel("$q_2$")
    ax.set_ylabel("$q_1$")
    ax.set_title("Best-Response Paths in Cournot Duopoly")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.legend(fontsize=8)
    report.add_figure(
        "figures/best-response-paths.png",
        "Best-response iteration converges to the Cournot Nash equilibrium",
        fig,
        description=(
            "Each path starts from a different output pair and moves by damped simultaneous "
            "best responses. The intersection of best-response curves is the Nash equilibrium."
        ),
    )

    fig2, ax2 = plt.subplots()
    for q0, path in paths.items():
        residuals = [fixed_point_residual(path[t], a, b, c) for t in range(len(path))]
        ax2.semilogy(residuals, marker="o", markersize=3, label=f"Start {q0}")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Fixed-point residual")
    ax2.set_title("Convergence Diagnostic")
    ax2.legend()
    report.add_figure(
        "figures/residuals.png",
        "Best-response residual falls toward zero",
        fig2,
        description=(
            "The residual measures the largest profitable correction implied by the best-response "
            "map. A Nash equilibrium has residual zero."
        ),
    )

    rows = []
    for q0, path in paths.items():
        q_final = path[-1]
        rows.append({
            "Initial q": str(q0),
            "Final q1": f"{q_final[0]:.4f}",
            "Final q2": f"{q_final[1]:.4f}",
            "Residual": f"{fixed_point_residual(q_final, a, b, c):.2e}",
        })
    rows.append({
        "Initial q": "Closed form",
        "Final q1": f"{q_star:.4f}",
        "Final q2": f"{q_star:.4f}",
        "Residual": f"{fixed_point_residual(np.array([q_star, q_star]), a, b, c):.2e}",
    })
    report.add_table(
        "tables/convergence-summary.csv",
        "Convergence Summary",
        pd.DataFrame(rows),
    )

    report.add_table(
        "tables/equilibrium-outcomes.csv",
        "Cournot Equilibrium Outcomes",
        pd.DataFrame([{
            "q1": f"{q_star:.4f}",
            "q2": f"{q_star:.4f}",
            "Price": f"{price_star:.4f}",
            "Profit per firm": f"{profit_star:.4f}",
        }]),
    )

    report.add_takeaway(
        "Best-response iteration turns a Nash equilibrium problem into a fixed-point problem. "
        "For this Cournot game the map is well behaved, so low-code iteration is enough. The "
        "important habit is to report a residual: convergence of the plotted path is not the "
        "same thing as verifying that no player still wants to deviate."
    )

    report.add_references([
        "[Cournot, A. A. (1838/1897). *Researches into the Mathematical Principles of the Theory of Wealth*. English translation.](https://openlibrary.org/books/OL5428468M/Researches_into_the_mathematical_principles_of_the_theory_of_wealth_1838.)",
        "[Fudenberg, D. and Levine, D. K. (1998). *The Theory of Learning in Games*. MIT Press.](https://mitpress.mit.edu/9780262061940/the-theory-of-learning-in-games/)",
        "[Fudenberg, D. and Tirole, J. (1991). *Game Theory*. MIT Press.](https://mitpress.mit.edu/9780262061414/game-theory/)",
        "[Vives, X. (1999). *Oligopoly Pricing: Old Ideas and New Tools*. MIT Press.](https://mitpress.mit.edu/9780262720403/oligopoly-pricing/)",
    ])

    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
