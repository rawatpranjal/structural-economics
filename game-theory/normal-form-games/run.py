#!/usr/bin/env python3
"""Normal-form games and Nash equilibrium checks.

Solves small finite games with direct enumeration and 2x2 indifference
conditions. No external game solver is used.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


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


def expected_payoff(
    payoffs: np.ndarray,
    row_prob_action0: float,
    col_prob_action0: float,
) -> float:
    """Expected payoff when both players mix over two actions."""
    row_mix = np.array([row_prob_action0, 1.0 - row_prob_action0])
    col_mix = np.array([col_prob_action0, 1.0 - col_prob_action0])
    return float(row_mix @ payoffs @ col_mix)


def main() -> None:
    games = {
        "Prisoner's Dilemma": {
            "row": np.array([[-1, -3], [0, -2]], dtype=float),
            "col": np.array([[-1, 0], [-3, -2]], dtype=float),
            "actions": (["Cooperate", "Defect"], ["Cooperate", "Defect"]),
        },
        "Matching Pennies": {
            "row": np.array([[1, -1], [-1, 1]], dtype=float),
            "col": np.array([[-1, 1], [1, -1]], dtype=float),
            "actions": (["Heads", "Tails"], ["Heads", "Tails"]),
        },
        "Battle of the Sexes": {
            "row": np.array([[3, 0], [0, 2]], dtype=float),
            "col": np.array([[2, 0], [0, 3]], dtype=float),
            "actions": (["Opera", "Football"], ["Opera", "Football"]),
        },
        "Stag Hunt": {
            "row": np.array([[4, 0], [3, 2]], dtype=float),
            "col": np.array([[4, 3], [0, 2]], dtype=float),
            "actions": (["Stag", "Hare"], ["Stag", "Hare"]),
        },
    }

    rows = []
    for name, game in games.items():
        pure = find_pure_nash(game["row"], game["col"])
        mixed = mixed_nash_2x2(game["row"], game["col"])
        mixed_text = "None"
        residual_text = ""
        if mixed is not None:
            p, q, residual = mixed
            mixed_text = f"p={p:.3f}, q={q:.3f}"
            residual_text = f"{residual:.1e}"

        rows.append({
            "Game": name,
            "Pure Nash": format_equilibria(pure, game["actions"]),
            "Interior mixed Nash": mixed_text,
            "Indifference residual": residual_text,
        })

    setup_style()
    report = ModelReport(
        "Normal-Form Games",
        "Nash equilibria by enumeration and 2x2 indifference conditions.",
    )

    report.add_overview(
        "A normal-form game lists each player's actions and payoffs for every action profile. "
        "For small games, the cleanest computational method is not a black-box solver: enumerate "
        "profiles, check best responses, and solve 2x2 mixed equilibria by making each player "
        "indifferent between the actions used with positive probability."
    )

    report.add_equations(r"""
**Pure Nash equilibrium:** action profile $(i^*, j^*)$ satisfies
$$
u_1(i^*, j^*) \geq u_1(i, j^*) \quad \forall i,
\qquad
u_2(i^*, j^*) \geq u_2(i^*, j) \quad \forall j.
$$

**2x2 mixed Nash equilibrium:** if the row player plays action 0 with probability $p$
and the column player plays action 0 with probability $q$, then an interior mixed equilibrium
solves the two indifference equations:
$$
u_1(0, q) = u_1(1, q),
\qquad
u_2(p, 0) = u_2(p, 1).
$$
""")

    report.add_model_setup(
        "The examples are four canonical 2x2 games: Prisoner's Dilemma, Matching Pennies, "
        "Battle of the Sexes, and Stag Hunt. The payoff matrices are hard-coded in the script "
        "so the equilibrium checks are visible and easy to audit."
    )

    report.add_solution_method(
        "**Pure equilibria:** loop over all cells and check whether both players are best "
        "responding.\n\n"
        "**Mixed equilibria:** solve the two linear indifference equations for 2x2 games, then "
        "verify that the mixing probabilities lie in $[0, 1]$ and report the indifference residual."
    )

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ax, (name, game) in zip(axes.ravel(), games.items()):
        total_payoffs = game["row"] + game["col"]
        im = ax.imshow(total_payoffs, cmap="Blues", vmin=-4, vmax=8)
        actions = game["actions"]
        ax.set_title(name)
        ax.set_xticks([0, 1], labels=actions[1])
        ax.set_yticks([0, 1], labels=actions[0])
        for i in range(2):
            for j in range(2):
                label = f"{game['row'][i, j]:.0f}, {game['col'][i, j]:.0f}"
                ax.text(j, i, label, ha="center", va="center", color="black")
        for i, j in find_pure_nash(game["row"], game["col"]):
            ax.scatter(j, i, marker="s", s=900, facecolors="none", edgecolors="crimson", linewidths=2.5)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="Total payoff")
    report.add_figure(
        "figures/payoff-matrices.png",
        "Payoff matrices with pure Nash equilibria outlined",
        fig,
        description=(
            "The red boxes mark pure Nash equilibria. Matching Pennies has no red box because "
            "every pure profile gives one player a profitable deviation."
        ),
    )

    p_grid = np.linspace(0, 1, 200)
    q_grid = np.linspace(0, 1, 200)
    mp = games["Matching Pennies"]
    row_diff = [
        (mp["row"][0] @ np.array([q, 1.0 - q])) - (mp["row"][1] @ np.array([q, 1.0 - q]))
        for q in q_grid
    ]
    col_diff = [
        (np.array([p, 1.0 - p]) @ mp["col"][:, 0]) - (np.array([p, 1.0 - p]) @ mp["col"][:, 1])
        for p in p_grid
    ]

    fig2, ax2 = plt.subplots()
    ax2.plot(q_grid, row_diff, label="Row payoff difference: Heads minus Tails")
    ax2.plot(p_grid, col_diff, label="Column payoff difference: Heads minus Tails")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.axvline(0.5, color="gray", linestyle=":", linewidth=1.5)
    ax2.set_xlabel("Opponent probability of Heads")
    ax2.set_ylabel("Payoff difference")
    ax2.set_title("Matching Pennies: Indifference at the Mixed Equilibrium")
    ax2.legend()
    report.add_figure(
        "figures/mixed-indifference.png",
        "Mixed equilibrium makes both players indifferent",
        fig2,
        description=(
            "At probability 0.5, each player's two pure actions have the same expected payoff. "
            "That indifference is what supports randomization."
        ),
    )

    df_games = pd.DataFrame(rows)
    report.add_table(
        "tables/equilibrium-summary.csv",
        "Equilibrium Summary",
        df_games,
        description="The residual column checks the mixed-equilibrium indifference equations.",
    )

    mixed_payoffs = []
    for name, game in games.items():
        mixed = mixed_nash_2x2(game["row"], game["col"])
        if mixed is None:
            continue
        p, q, _ = mixed
        mixed_payoffs.append({
            "Game": name,
            "Row payoff": f"{expected_payoff(game['row'], p, q):.3f}",
            "Column payoff": f"{expected_payoff(game['col'], p, q):.3f}",
            "p": f"{p:.3f}",
            "q": f"{q:.3f}",
        })
    report.add_table(
        "tables/mixed-payoffs.csv",
        "Expected Payoffs at Interior Mixed Equilibria",
        pd.DataFrame(mixed_payoffs),
    )

    report.add_takeaway(
        "The computational lesson is simple: for small finite games, equilibrium is a set of "
        "inequality and indifference checks. Pure Nash equilibria are found by checking unilateral "
        "deviations cell by cell. Mixed equilibria are found by making the opponent indifferent. "
        "This gives a transparent baseline before using heavier fixed-point or dynamic-game methods."
    )

    report.add_references([
        "Nash, J. (1950). Equilibrium Points in N-Person Games. *Proceedings of the National Academy of Sciences*, 36(1).",
        "Osborne, M. and Rubinstein, A. (1994). *A Course in Game Theory*. MIT Press.",
    ])

    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
