#!/usr/bin/env python3
"""Finite strategic games and Nash equilibrium checks.

Solves small finite games with direct enumeration and 2x2 indifference
conditions. No external game solver is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


def find_pure_nash(row_payoffs: np.ndarray, col_payoffs: np.ndarray) -> list[tuple[int, int]]:
    """Return all pure Nash equilibria in a two-player normal-form game."""
    n_rows, n_cols = row_payoffs.shape
    equilibria: list[tuple[int, int]] = []

    for i in range(n_rows):
        for j in range(n_cols):
            row_best = row_payoffs[i, j] == np.max(row_payoffs[:, j])
            col_best = col_payoffs[i, j] == np.max(col_payoffs[i, :])
            if row_best and col_best:
                equilibria.append((i, j))

    return equilibria


def unilateral_deviation_gains(row_payoffs: np.ndarray, col_payoffs: np.ndarray) -> np.ndarray:
    """Largest profitable one-player deviation at each pure action profile."""
    n_rows, n_cols = row_payoffs.shape
    gains = np.zeros((n_rows, n_cols), dtype=float)

    for i in range(n_rows):
        for j in range(n_cols):
            row_gain = np.max(row_payoffs[:, j]) - row_payoffs[i, j]
            col_gain = np.max(col_payoffs[i, :]) - col_payoffs[i, j]
            gains[i, j] = max(row_gain, col_gain)

    return gains


def mixed_nash_2x2(
    row_payoffs: np.ndarray,
    col_payoffs: np.ndarray,
) -> tuple[float, float, float] | None:
    """Solve the interior mixed Nash equilibrium of a 2x2 game if it exists.

    Returns:
        A tuple `(p, q, residual)`, where `p` is the row player's probability
        of action 0, `q` is the column player's probability of action 0, and
        `residual` is the largest absolute indifference error.
    """
    a = row_payoffs
    b = col_payoffs

    denom_q = a[0, 0] - a[1, 0] - a[0, 1] + a[1, 1]
    denom_p = b[0, 0] - b[0, 1] - b[1, 0] + b[1, 1]
    if abs(denom_q) < 1e-12 or abs(denom_p) < 1e-12:
        return None

    q = (a[1, 1] - a[0, 1]) / denom_q
    p = (b[1, 1] - b[1, 0]) / denom_p
    if not (0.0 <= p <= 1.0 and 0.0 <= q <= 1.0):
        return None

    row_action_payoffs = row_payoffs @ np.array([q, 1.0 - q])
    col_action_payoffs = np.array([p, 1.0 - p]) @ col_payoffs
    residual = max(
        abs(row_action_payoffs[0] - row_action_payoffs[1]),
        abs(col_action_payoffs[0] - col_action_payoffs[1]),
    )
    return float(p), float(q), float(residual)


def format_equilibria(equilibria: list[tuple[int, int]], actions: tuple[list[str], list[str]]) -> str:
    """Format equilibrium action profiles for the report table."""
    if not equilibria:
        return "None"
    return ", ".join(f"({actions[0][i]}, {actions[1][j]})" for i, j in equilibria)


def main() -> None:
    games = {
        "Prisoner's Dilemma": {
            "row": np.array([[-1, -3], [0, -2]], dtype=float),
            "col": np.array([[-1, 0], [-3, -2]], dtype=float),
            "actions": (["Cooperate", "Defect"], ["Cooperate", "Defect"]),
            "pattern": "Defection is stable even though cooperation has higher joint payoff.",
        },
        "Matching Pennies": {
            "row": np.array([[1, -1], [-1, 1]], dtype=float),
            "col": np.array([[-1, 1], [1, -1]], dtype=float),
            "actions": (["Heads", "Tails"], ["Heads", "Tails"]),
            "pattern": "Any predictable pure action invites a profitable response.",
        },
        "Battle of the Sexes": {
            "row": np.array([[3, 0], [0, 2]], dtype=float),
            "col": np.array([[2, 0], [0, 3]], dtype=float),
            "actions": (["Opera", "Football"], ["Opera", "Football"]),
            "pattern": "Two conventions are stable; mixing balances conflicting preferred outcomes.",
        },
        "Stag Hunt": {
            "row": np.array([[4, 0], [3, 2]], dtype=float),
            "col": np.array([[4, 3], [0, 2]], dtype=float),
            "actions": (["Stag", "Hare"], ["Stag", "Hare"]),
            "pattern": "Safe and payoff-dominant conventions both satisfy no-deviation.",
        },
    }

    rows = []
    for name, game in games.items():
        pure = find_pure_nash(game["row"], game["col"])
        mixed = mixed_nash_2x2(game["row"], game["col"])
        mixed_text = "None"
        residual_text = "None"
        if mixed is not None:
            p, q, residual = mixed
            row_first = game["actions"][0][0]
            col_first = game["actions"][1][0]
            mixed_text = f"Pr(row {row_first})={p:.3f}; Pr(column {col_first})={q:.3f}"
            residual_text = f"{residual:.1e}"

        rows.append({
            "Game": name,
            "Pure Nash equilibria": format_equilibria(pure, game["actions"]),
            "Interior mixed equilibrium": mixed_text,
            "Indifference residual": residual_text,
            "Economic pattern": game["pattern"],
        })

    setup_style()

    max_gain = max(
        float(np.max(unilateral_deviation_gains(game["row"], game["col"])))
        for game in games.values()
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.8), constrained_layout=True)
    for ax, (name, game) in zip(axes.ravel(), games.items()):
        deviation_gains = unilateral_deviation_gains(game["row"], game["col"])
        im = ax.imshow(deviation_gains, cmap="YlOrRd", vmin=0.0, vmax=max_gain)
        actions = game["actions"]
        ax.set_title(name)
        ax.set_xticks([0, 1], labels=actions[1])
        ax.set_yticks([0, 1], labels=actions[0])
        for i in range(2):
            for j in range(2):
                label = f"{game['row'][i, j]:.0f}, {game['col'][i, j]:.0f}"
                ax.text(j, i, label, ha="center", va="center", color="black")
        for i, j in find_pure_nash(game["row"], game["col"]):
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="black",
                    linewidth=2.5,
                )
            )
            ax.text(j, i + 0.31, "Nash", ha="center", va="center", fontsize=8, fontweight="bold")
    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.78,
        label="Largest profitable unilateral deviation",
    )
    save_figure(fig, "figures/pure-deviation-gains.png", dpi=150)

    p_grid = np.linspace(0, 1, 200)
    q_grid = np.linspace(0, 1, 200)
    interior_games = [
        (name, game, mixed_nash_2x2(game["row"], game["col"]))
        for name, game in games.items()
        if mixed_nash_2x2(game["row"], game["col"]) is not None
    ]

    fig2, axes2 = plt.subplots(1, len(interior_games), figsize=(12, 3.8), constrained_layout=True)
    for k, (ax, (name, game, mixed)) in enumerate(zip(np.atleast_1d(axes2), interior_games)):
        assert mixed is not None
        p, q, _ = mixed
        row_payoff_diff = [
            (game["row"][0] @ np.array([q0, 1.0 - q0]))
            - (game["row"][1] @ np.array([q0, 1.0 - q0]))
            for q0 in q_grid
        ]
        col_payoff_diff = [
            (np.array([p0, 1.0 - p0]) @ game["col"][:, 0])
            - (np.array([p0, 1.0 - p0]) @ game["col"][:, 1])
            for p0 in p_grid
        ]
        row_label = f"Row: {game['actions'][0][0]} - {game['actions'][0][1]}"
        col_label = f"Column: {game['actions'][1][0]} - {game['actions'][1][1]}"
        row_line = ax.plot(q_grid, row_payoff_diff, linewidth=2.0, label=row_label)[0]
        col_line = ax.plot(p_grid, col_payoff_diff, linewidth=2.0, label=col_label)[0]
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.axvline(q, color=row_line.get_color(), linestyle=":", linewidth=1.5)
        ax.axvline(p, color=col_line.get_color(), linestyle="--", linewidth=1.5)
        ax.scatter([q], [0.0], color=row_line.get_color(), zorder=5)
        ax.scatter([p], [0.0], color=col_line.get_color(), zorder=5)
        ax.set_title(f"{name}\nq={q:.2f}, p={p:.2f}")
        ax.set_xlabel("Opponent probability of first action")
        if k == 0:
            ax.set_ylabel("Expected payoff difference")
        ax.legend(fontsize=7)
    save_figure(fig2, "figures/mixed-indifference.png", dpi=150)

    df_games = pd.DataFrame(rows)
    Path("tables").mkdir(parents=True, exist_ok=True)
    df_games.to_csv("tables/equilibrium-summary.csv", index=False)

    save_thumbnail("figures/pure-deviation-gains.png", "figures/thumb.png")
    print(f"Done: 2 figures + 1 table")


if __name__ == "__main__":
    main()
