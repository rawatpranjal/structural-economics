#!/usr/bin/env python3
"""Normal-form games and Nash equilibrium checks.

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
            "pattern": "Dominance: defection is the unique stable profile.",
        },
        "Matching Pennies": {
            "row": np.array([[1, -1], [-1, 1]], dtype=float),
            "col": np.array([[-1, 1], [1, -1]], dtype=float),
            "actions": (["Heads", "Tails"], ["Heads", "Tails"]),
            "pattern": "No pure equilibrium; mixing removes predictable play.",
        },
        "Battle of the Sexes": {
            "row": np.array([[3, 0], [0, 2]], dtype=float),
            "col": np.array([[2, 0], [0, 3]], dtype=float),
            "actions": (["Opera", "Football"], ["Opera", "Football"]),
            "pattern": "Two coordination equilibria plus one mixed conflict point.",
        },
        "Stag Hunt": {
            "row": np.array([[4, 0], [3, 2]], dtype=float),
            "col": np.array([[4, 3], [0, 2]], dtype=float),
            "actions": (["Stag", "Hare"], ["Stag", "Hare"]),
            "pattern": "Two coordination equilibria, one payoff dominant.",
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
            "Equilibrium pattern": game["pattern"],
        })

    setup_style()
    report = ModelReport(
        "Normal-Form Games and Nash Equilibrium Checks",
        "Pure profiles, mixed supports, and unilateral-deviation residuals.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A normal-form game is the payoff table behind many richer models. Before adding "
        "states, prices, or private information, Nash equilibrium is a set of "
        "no-profitable-deviation restrictions on that table. The games here are small "
        "enough that those restrictions can be inspected directly.\n\n"
        "The point is not to showcase a solver. Pure equilibria come from checking every "
        "action profile. Interior mixed equilibria in 2x2 games come from the two "
        "indifference equations that make randomization optimal. The same logic is the "
        "static baseline for [Cournot best-response dynamics](../static-games/) and the "
        "exact benchmark that [quantal response equilibrium](../quantal-response-equilibrium/) "
        "softens into noisy best responses."
    )

    report.add_equations(r"""
There are two players. The row player has actions $i \in I$, the column player
has actions $j \in J$, and the payoff matrices are $A$ for the row player and
$B$ for the column player. At pure profile $(i,j)$, payoffs are $(A_{ij},B_{ij})$.

The row player's one-step deviation gain at $(i,j)$ is

$$
d_1(i,j)=\max_{i' \in I} A_{i'j}-A_{ij},
$$

and the column player's gain is

$$
d_2(i,j)=\max_{j' \in J} B_{ij'}-B_{ij}.
$$

A pure Nash equilibrium is a profile $(i^{*}, j^{*})$ with

$$
d_1(i^{*},j^{*})=d_2(i^{*},j^{*})=0,
$$

equivalently

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
residual is the maximum absolute gap in these two indifference equations.
""")

    report.add_model_setup(
        "Four canonical 2x2 games are used. Each payoff table is small enough that the "
        "economic tension is visible in the cells, and the equilibrium patterns differ "
        "enough to separate dominance, zero-sum mixing, and coordination.\n\n"
        "| Game | Actions | What the payoffs isolate |\n"
        "|---|---|---|\n"
        "| Prisoner's Dilemma | Cooperate/Defect | Individual incentives overturn the efficient profile. |\n"
        "| Matching Pennies | Heads/Tails | No pure action can be predictable in equilibrium. |\n"
        "| Battle of the Sexes | Opera/Football | Coordination is valuable, but players prefer different conventions. |\n"
        "| Stag Hunt | Stag/Hare | Safe and payoff-dominant coordination profiles coexist. |"
    )

    report.add_solution_method(
        "The computation is exact for these finite games. Enumeration handles pure "
        "profiles. The 2x2 mixed calculation solves the closed-form indifference system "
        "and then checks the candidate rather than trusting the formula mechanically.\n\n"
        "```text\n"
        "Algorithm: Nash checks for a two-player finite game\n"
        "Inputs: payoff matrices A, B and action labels I, J\n"
        "Outputs: pure Nash set E and, for 2x2 games, an interior mixed candidate\n\n"
        "1. For each pure profile (i,j), compute d1(i,j) and d2(i,j).\n"
        "2. Add (i,j) to E when max{d1(i,j), d2(i,j)} = 0.\n"
        "3. If the game is 2x2, solve the two linear indifference equations for p and q.\n"
        "4. Keep the mixed candidate only when p and q lie in [0,1].\n"
        "5. Recompute both expected-payoff gaps and report the largest absolute residual.\n"
        "```\n\n"
        "This residual is the diagnostic. A profile or mixed strategy is not interesting "
        "because an algorithm named it; it is interesting because no player can improve "
        "by changing only its own action."
    )

    report.add_results(
        "The first figure colors each pure action profile by the largest profitable "
        "unilateral deviation. A zero-deviation cell is a pure Nash equilibrium, so the "
        "black outlines are not decorative markers; they are the cells where the "
        "equilibrium inequalities bind. This also separates efficiency from equilibrium. "
        "In Prisoner's Dilemma, mutual cooperation has higher joint payoff than mutual "
        "defection, but is not stable against a one-player deviation."
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
        "Pure payoff tables colored by unilateral-deviation gain",
        fig,
    )

    report.add_results(
        "The mixed-strategy figure uses the closed-form indifference equations as the "
        "ground truth. Each curve is the expected payoff from the first action minus "
        "the expected payoff from the second action. A root is where the opponent's "
        "mix makes that player willing to randomize. Matching Pennies has the symmetric "
        "half-half root; Battle of the Sexes has asymmetric mixing because the players "
        "prefer different coordinated outcomes; Stag Hunt's mixed equilibrium is the "
        "knife-edge between the safe and payoff-dominant basins."
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
        "Exact mixed-equilibrium indifference roots",
        fig2,
    )

    df_games = pd.DataFrame(rows)
    report.add_table(
        "tables/equilibrium-summary.csv",
        "Equilibrium Summary by Game",
        df_games,
        description=(
            "The summary table reports the exact pure-equilibrium set and the interior "
            "mixed candidate when one exists. Residuals are numerical checks of the "
            "closed-form indifference equations."
        ),
    )

    mixed_payoffs = []
    for name, game in games.items():
        mixed = mixed_nash_2x2(game["row"], game["col"])
        if mixed is None:
            continue
        p, q, _ = mixed
        mixed_payoffs.append({
            "Game": name,
            "Row first-action probability p": f"{p:.3f}",
            "Column first-action probability q": f"{q:.3f}",
            "Row payoff": f"{expected_payoff(game['row'], p, q):.3f}",
            "Column payoff": f"{expected_payoff(game['col'], p, q):.3f}",
        })
    report.add_table(
        "tables/mixed-payoffs.csv",
        "Expected Payoffs at Interior Mixed Equilibria",
        pd.DataFrame(mixed_payoffs),
        description=(
            "Expected payoffs at the mixed equilibria are included because the "
            "probabilities alone can hide the economic tradeoff. In zero-sum Matching "
            "Pennies both players get zero; in the coordination games the mixed point "
            "is worse than successful coordination."
        ),
    )

    report.add_takeaway(
        "For finite static games, Nash equilibrium is best read as a residual condition. "
        "Pure equilibria are zero-deviation cells in the payoff table. Interior mixed "
        "equilibria choose probabilities that make opponents indifferent across the "
        "actions they use. This direct calculation is the benchmark before moving to "
        "fixed-point iteration, noisy response, or dynamic games where the same "
        "no-deviation logic is harder to see."
    )

    report.add_references([
        "[Nash, J. (1950). Equilibrium Points in N-Person Games. *Proceedings of the National Academy of Sciences*, 36(1), 48-49.](https://doi.org/10.1073/pnas.36.1.48)",
        "[Osborne, M. and Rubinstein, A. (1994). *A Course in Game Theory*. MIT Press.](https://mitpress.mit.edu/9780262650403/a-course-in-game-theory)",
        "[Lemke, C. E. and Howson, J. T. (1964). Equilibrium Points of Bimatrix Games. *SIAM Journal on Applied Mathematics*, 12(2), 413-423.](https://doi.org/10.1137/0112033)",
    ])

    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
