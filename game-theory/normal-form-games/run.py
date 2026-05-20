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
    report = ModelReport(
        "Finite Strategic Games and Nash Equilibrium Checks",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Strategic settings often turn on unilateral incentives. An outcome is stable "
        "only when each player is content with its own action.\n\n"
        "A normal-form game stores those incentives in a payoff table. Each cell lists "
        "the row and column payoffs for one action profile.\n\n"
        "The computation asks two questions. Which cells have zero profitable "
        "one-player deviations? In 2x2 games, which probabilities make both players "
        "indifferent over the actions they use?"
    )

    report.add_equations(r"""
A finite two-player game has a row player with actions $i \in I$ and a column
player with actions $j \in J$. The matrices $A$ and $B$ record row and column
payoffs. At pure profile $(i,j)$, the players receive $(A_{ij},B_{ij})$.

The row player's one-step deviation gain at $(i,j)$ is

$$
d_1(i,j)=\max_{i' \in I} A_{i'j}-A_{ij},
$$

and the column player's gain is

$$
d_2(i,j)=\max_{j' \in J} B_{ij'}-B_{ij}.
$$

The combined deviation gain at $(i,j)$ is the larger of the two,

$$
d(i,j)=\max\lbrace d_1(i,j), d_2(i,j) \rbrace.
$$

The heat maps color each cell by $d(i,j)$, and the pseudocode tests
$d(i,j)=0$.

A pure Nash equilibrium is a profile $(i^{*}, j^{*})$ with

$$
d_1(i^{*},j^{*})=d_2(i^{*},j^{*})=0,
$$

Equivalently, the two best-response inequalities are

$$
A_{i^{*}j^{*}} \geq A_{ij^{*}} \quad \forall i \in I,
\qquad
B_{i^{*}j^{*}} \geq B_{i^{*}j} \quad \forall j \in J.
$$

For a 2x2 game, let the row player use mixed strategy $x=(p,1-p)$ and the
column player use $y=(q,1-q)$. An interior mixed equilibrium requires both
players to be indifferent across the actions used with positive probability:

$$
A_{11}q + A_{12}(1-q) = A_{21}q + A_{22}(1-q),
\qquad
B_{11}p + B_{21}(1-p) = B_{12}p + B_{22}(1-p).
$$

The candidate is an equilibrium only if $p,q \in [0,1]$. The reported mixed
residual is the maximum absolute gap left in these two indifference equations.
""")

    report.add_model_setup(
        "Four 2x2 games make the checks concrete. Prisoner's Dilemma isolates private "
        "incentives against joint surplus. Matching Pennies has no pure equilibrium. "
        "Battle of the Sexes has two conventions and conflicting preferences. Stag Hunt "
        "has a safe action and a payoff-dominant convention.\n\n"
        "| Game | Actions | What the payoffs isolate |\n"
        "|---|---|---|\n"
        "| Prisoner's Dilemma | Cooperate/Defect | Individual incentives overturn the efficient profile. |\n"
        "| Matching Pennies | Heads/Tails | No pure action can be predictable in equilibrium. |\n"
        "| Battle of the Sexes | Opera/Football | Coordination is valuable, but players prefer different conventions. |\n"
        "| Stag Hunt | Stag/Hare | Safe and payoff-dominant coordination profiles coexist. |"
    )

    report.add_solution_method(
        "Equilibrium is a finite set of inequalities. The code computes deviation gains "
        "at every pure profile. For each 2x2 game, it also solves the two linear "
        "indifference equations for p and q.\n\n"
        "```text\n"
        "Algorithm: Nash checks for a two-player finite game\n"
        "Inputs: payoff matrices A, B and action labels I, J\n"
        "Outputs: pure Nash set E and, for 2x2 games, an interior mixed candidate\n\n"
        "1. For each pure profile (i,j), compute d1(i,j) and d2(i,j).\n"
        "2. Add (i,j) to E when max{d1(i,j), d2(i,j)} = 0.\n"
        "3. For each 2x2 game, solve the two indifference equations for p and q.\n"
        "4. Keep the mixed candidate only when p and q lie in [0,1].\n"
        "5. Recompute both expected-payoff gaps and report the largest absolute residual.\n"
        "```\n\n"
        "The residual checks the mixed calculation. Pure profiles pass when both "
        "deviation gains equal zero. Mixed profiles pass when both actions used in the "
        "mixture have equal expected payoffs."
    )

    report.add_results(
        "The heat maps color each payoff table by the largest one-player deviation "
        "gain. Warmer cells have larger gains from switching action. A black outline "
        "marks a zero-deviation cell. In Prisoner's Dilemma, mutual defection is stable "
        "even though mutual cooperation gives more total payoff."
    )

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
    report.add_figure(
        "figures/pure-deviation-gains.png",
        "Payoff tables colored by profitable deviation gains",
        fig,
    )

    report.add_results(
        "The mixed-strategy panels show the payoff differences behind randomization. "
        "Each curve subtracts the second-action payoff from the first-action payoff. A "
        "root gives the opponent probability that makes the player willing to mix. "
        "Matching Pennies lands at half-half. Battle of the Sexes gives asymmetric "
        "probabilities. Stag Hunt gives a threshold between safe and payoff-dominant "
        "coordination."
    )

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
    report.add_figure(
        "figures/mixed-indifference.png",
        "Mixed-strategy indifference roots in 2x2 games",
        fig2,
    )

    df_games = pd.DataFrame(rows)
    report.add_table(
        "tables/equilibrium-summary.csv",
        "Equilibrium Summary by Game",
        df_games,
        description=(
            "The summary table lists the equilibria from the same checks. Pure entries "
            "are zero-deviation cells. Mixed entries give the interior probability pair "
            "and the largest indifference residual."
        ),
    )

    report.add_takeaway(
        "Finite normal-form games make Nash equilibrium directly checkable. Enumeration "
        "finds pure equilibria by testing profitable one-player deviations. The 2x2 "
        "mixed check chooses probabilities that erase payoff gaps within each player's "
        "support."
    )

    report.add_references([
        "[Nash, J. (1950). Equilibrium Points in N-Person Games. *Proceedings of the National Academy of Sciences*, 36(1), 48-49.](https://doi.org/10.1073/pnas.36.1.48)",
        "[Osborne, M. and Rubinstein, A. (1994). *A Course in Game Theory*. MIT Press.](https://mitpress.mit.edu/9780262650403/a-course-in-game-theory)",
    ])

    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
