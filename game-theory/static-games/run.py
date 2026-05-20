#!/usr/bin/env python3
"""Cournot oligopoly and best-response dynamics."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


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
    paths = {
        q0: iterate_best_responses(q0, a, b, c, steps=steps, damping=damping)
        for q0 in starts
    }

    setup_style()

    q_grid = np.linspace(0.0, 8.0, 240)
    br = cournot_best_response(q_grid, a, b, c)

    fig1, ax1 = plt.subplots(figsize=(7, 6))
    ax1.plot(q_grid, br, linewidth=2.3, label="$BR_1(q_2)$")
    ax1.plot(br, q_grid, linewidth=2.3, label="$BR_2(q_1)$")
    ax1.scatter(q_star, q_star, color="black", s=60, zorder=5, label=f"Nash q={q_star:.2f}")
    for q0, path in paths.items():
        ax1.plot(path[:, 1], path[:, 0], marker="o", markersize=2.5, alpha=0.65, label=f"Start {q0}")
    ax1.set_xlabel("$q_2$")
    ax1.set_ylabel("$q_1$")
    ax1.set_title("Cournot Best Responses and Iteration Paths")
    ax1.set_xlim(0.0, 8.0)
    ax1.set_ylim(0.0, 8.0)
    ax1.legend(fontsize=8)
    save_figure(fig1, "figures/cournot-best-response.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for q0, path in paths.items():
        residuals = [fixed_point_residual(path[t], a, b, c) for t in range(len(path))]
        ax2.semilogy(residuals, marker="o", markersize=3, label=f"Start {q0}")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Fixed-point residual")
    ax2.set_title("Best-Response Residual")
    ax2.legend()
    save_figure(fig2, "figures/residuals.png", dpi=150)

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

    Path("tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv("tables/convergence-summary.csv", index=False)

    save_thumbnail("figures/cournot-best-response.png", "figures/thumb.png")
    print("Done: figures/cournot-best-response.png, figures/residuals.png, tables/convergence-summary.csv")


if __name__ == "__main__":
    main()
